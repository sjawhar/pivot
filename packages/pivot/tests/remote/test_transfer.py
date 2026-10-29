from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import xxhash

from pivot import exceptions, project
from pivot.remote import config as remote_config
from pivot.remote import keys
from pivot.remote import storage as remote_mod
from pivot.remote import sync as transfer
from pivot.storage import cache as cache_mod
from pivot.storage import listing, lock
from pivot.storage import state as state_mod
from pivot.types import (
    ArtifactIdentity,
    DirManifestEntry,
    FileHash,
    HashInfo,
    LockData,
    RemoteStatus,
    TransferResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture
    from types_aiobotocore_s3 import S3Client

    from pivot.registry import RegistryStageInfo


def _helper_make_all_stages(*stage_names: str) -> dict[str, RegistryStageInfo]:
    """Create minimal all_stages dict with only the fields get_target_hashes reads."""
    from pivot.registry import RegistryStageInfo

    return {
        name: RegistryStageInfo(  # pyright: ignore[reportCallIssue] - partial for test
            state_dir=None,
            outs=[],
            input_paths={},
        )
        for name in stage_names
    }


def _helper_write_lock(
    state_dir: Path,
    stage_name: str,
    *,
    outs: dict[str | None, HashInfo] | None = None,
    deps: dict[str, HashInfo] | None = None,
    directories: tuple[listing.HashedDirectory, ...] = (),
) -> lock.StageLock:
    """Write a lock file, with the committed listing of each directory output it records."""
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    stage_lock.write(
        LockData(
            code_manifest={},
            params={},
            dep_hashes={ArtifactIdentity(name, None): h for name, h in (deps or {}).items()},
            output_hashes={ArtifactIdentity(stage_name, key): h for key, h in (outs or {}).items()},
        ),
        hashed_dirs={directory["record"]["hash"]: directory for directory in directories},
    )
    return stage_lock


def _helper_hashed_dir(*files: tuple[str, str, int]) -> listing.HashedDirectory:
    """Build a directory record and the listing bytes its hash names."""
    members = listing.DirListing(
        files=[
            DirManifestEntry(relpath=relpath, hash=h, size=size, isexec=False)
            for relpath, h, size in files
        ],
        empty_dirs=[],
    )
    data = listing.encode(members)
    return listing.HashedDirectory(
        record=listing.record_of(members, data), listing=members, data=data
    )


def _helper_cache_object(files_dir: Path, content: bytes) -> str:
    """Put content in a local cache at its xxhash64 key and return the key."""
    content_hash = xxhash.xxh64(content).hexdigest()
    cache_path = cache_mod.get_cache_path(files_dir, content_hash)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    return content_hash


async def _helper_remote_object(
    s3_remote: remote_mod.S3Remote, client: S3Client, object_hash: str
) -> bytes:
    """Read an object pushed to the moto remote by cache key."""
    response = await client.get_object(
        Bucket=s3_remote.bucket, Key=keys.hash_to_key(s3_remote.prefix, object_hash)
    )
    return await response["Body"].read()


async def _helper_put_remote_object(
    s3_remote: remote_mod.S3Remote, client: S3Client, object_hash: str, content: bytes
) -> None:
    """Put an object on the moto remote by cache key."""
    await client.put_object(
        Bucket=s3_remote.bucket,
        Key=keys.hash_to_key(s3_remote.prefix, object_hash),
        Body=content,
    )


# -----------------------------------------------------------------------------
# Local Cache Hash Scanning Tests
# -----------------------------------------------------------------------------


def test_get_local_cache_hashes_empty(tmp_path: Path) -> None:
    """Empty cache returns empty set."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    result = transfer.get_local_cache_hashes(cache_dir)
    assert result == set()


def test_get_local_cache_hashes_no_files_dir(tmp_path: Path) -> None:
    """Missing files directory returns empty set."""
    cache_dir = tmp_path / "cache"

    result = transfer.get_local_cache_hashes(cache_dir)
    assert result == set()


def test_get_local_cache_hashes(tmp_path: Path) -> None:
    """Scans cache files directory and extracts hashes."""
    cache_dir = tmp_path / "cache"
    files_dir = cache_dir / "files"

    # Create cache structure: files/XX/YYYYYYYY...
    hash1 = "ab" + "c" * 14  # 16 chars total (xxhash64)
    hash2 = "de" + "f" * 14
    hash3 = "12" + "3" * 14

    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    (files_dir / "de").mkdir(parents=True)
    (files_dir / "de" / ("f" * 14)).write_text("content2")

    (files_dir / "12").mkdir(parents=True)
    (files_dir / "12" / ("3" * 14)).write_text("content3")

    result = transfer.get_local_cache_hashes(cache_dir)
    assert result == {hash1, hash2, hash3}


def test_get_local_cache_hashes_ignores_invalid_structure(tmp_path: Path) -> None:
    """Ignores files not matching expected hash structure."""
    cache_dir = tmp_path / "cache"
    files_dir = cache_dir / "files"

    # Valid hash
    valid_hash = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("valid")

    # Invalid: prefix too long
    (files_dir / "abc").mkdir(parents=True)
    (files_dir / "abc" / "def").write_text("invalid")

    # Invalid: wrong total length
    (files_dir / "xy").mkdir(parents=True)
    (files_dir / "xy" / "short").write_text("invalid")

    result = transfer.get_local_cache_hashes(cache_dir)
    assert result == {valid_hash}


# -----------------------------------------------------------------------------
# Target Output Hash Resolution Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def lock_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    # Lock files are at .pivot/stages/, cache at .pivot/cache/
    (tmp_path / ".pivot" / "stages").mkdir(parents=True)
    (tmp_path / ".pivot" / "cache").mkdir(parents=True)
    return tmp_path


def test_get_target_hashes_stage_without_lock_returns_empty(lock_project: Path) -> None:
    """A known stage with no lock has no cache objects to transfer."""
    state_dir = lock_project / ".pivot"

    result = transfer.get_target_hashes(
        ["missing_stage"],
        state_dir,
        all_stages=_helper_make_all_stages("missing_stage"),
        cache_dir=state_dir / "cache",
    )
    assert result == set()


def test_get_target_hashes_file_output(lock_project: Path) -> None:
    """A target stage's file output names its cache object."""
    state_dir = lock_project / ".pivot"

    _helper_write_lock(state_dir, "my_stage", outs={None: FileHash(hash="abc123def4567800")})

    result = transfer.get_target_hashes(
        ["my_stage"],
        state_dir,
        all_stages=_helper_make_all_stages("my_stage"),
        cache_dir=state_dir / "cache",
    )
    assert result == {"abc123def4567800"}


def test_get_target_hashes_directory_output_stores_listing_object(lock_project: Path) -> None:
    """A directory output names its listing object and listed files for push."""
    state_dir = lock_project / ".pivot"
    files_dir = state_dir / "cache" / "files"
    directory = _helper_hashed_dir(
        ("file1.txt", "f11e1a5b12345670", 100),
        ("file2.txt", "f11e2a5b23456780", 200),
    )

    _helper_write_lock(
        state_dir,
        "my_stage",
        outs={"output_dir": directory["record"]},
        directories=(directory,),
    )

    result = transfer.get_target_hashes(
        ["my_stage"],
        state_dir,
        all_stages=_helper_make_all_stages("my_stage"),
        cache_dir=state_dir / "cache",
    )

    assert result == {directory["record"]["hash"], "f11e1a5b12345670", "f11e2a5b23456780"}
    assert cache_mod.cached_listing(files_dir, directory["record"]) == directory["listing"], (
        "Push should be able to send the listing object keyed by the directory hash"
    )


def test_get_target_hashes_multiple_stage_outputs(lock_project: Path) -> None:
    """Collects target hashes from multiple stages."""
    state_dir = lock_project / ".pivot"

    for i, stage in enumerate(["stage_a", "stage_b"]):
        _helper_write_lock(state_dir, stage, outs={None: FileHash(hash=f"a{i}{'0' * 14}")})

    result = transfer.get_target_hashes(
        ["stage_a", "stage_b"],
        state_dir,
        all_stages=_helper_make_all_stages("stage_a", "stage_b"),
        cache_dir=state_dir / "cache",
    )
    assert result == {"a0" + "0" * 14, "a1" + "0" * 14}


# -----------------------------------------------------------------------------
# Target Dependency Hash Resolution Tests
# -----------------------------------------------------------------------------


def test_get_target_hashes_include_deps_file_outputs(lock_project: Path) -> None:
    """A target stage's cached upstream file outputs are included."""
    state_dir = lock_project / ".pivot"

    _helper_write_lock(
        state_dir,
        "my_stage",
        deps={
            "upstream_a": FileHash(hash="de01a5b123456700"),
            "upstream_b": FileHash(hash="de02a5b123456700"),
        },
    )

    result = transfer.get_target_hashes(
        ["my_stage"],
        state_dir,
        include_deps=True,
        all_stages=_helper_make_all_stages("my_stage", "upstream_a", "upstream_b"),
        cache_dir=state_dir / "cache",
    )
    assert result == {"de01a5b123456700", "de02a5b123456700"}


def test_get_target_hashes_include_deps_directory_listing_and_files(lock_project: Path) -> None:
    """An upstream directory dependency names its listing object and listed files."""
    state_dir = lock_project / ".pivot"
    directory = _helper_hashed_dir(
        ("a.txt", "af11ea5b12345600", 10),
        ("b.txt", "bf11ea5b12345600", 20),
    )
    _helper_write_lock(
        state_dir,
        "upstream",
        outs={None: directory["record"]},
        directories=(directory,),
    )
    _helper_write_lock(state_dir, "my_stage", deps={"upstream": directory["record"]})

    result = transfer.get_target_hashes(
        ["my_stage"],
        state_dir,
        include_deps=True,
        all_stages=_helper_make_all_stages("my_stage", "upstream"),
        cache_dir=state_dir / "cache",
    )
    assert result == {directory["record"]["hash"], "af11ea5b12345600", "bf11ea5b12345600"}


def test_get_target_hashes_directory_without_listing_names_listing_object(
    lock_project: Path,
) -> None:
    """A directory whose listing is not local is first transferred as its listing object."""
    state_dir = lock_project / ".pivot"
    directory = _helper_hashed_dir(("a.txt", "af11ea5b12345600", 10))
    stage_lock = _helper_write_lock(
        state_dir,
        "upstream",
        outs={None: directory["record"]},
        directories=(directory,),
    )
    stage_lock.output_listing_path(ArtifactIdentity("upstream", None)).unlink()
    _helper_write_lock(state_dir, "my_stage", deps={"upstream": directory["record"]})

    result = transfer.get_target_hashes(
        ["my_stage"],
        state_dir,
        include_deps=True,
        all_stages=_helper_make_all_stages("my_stage", "upstream"),
        cache_dir=state_dir / "cache",
    )
    assert result == {directory["record"]["hash"]}


def test_get_target_hashes_include_deps_no_lock(lock_project: Path) -> None:
    """A target stage with no lock has no dependency objects to transfer."""
    state_dir = lock_project / ".pivot"

    result = transfer.get_target_hashes(
        ["missing_stage"],
        state_dir,
        include_deps=True,
        all_stages=_helper_make_all_stages("missing_stage"),
        cache_dir=state_dir / "cache",
    )
    assert result == set()


# -----------------------------------------------------------------------------
# Compare Status Tests
# -----------------------------------------------------------------------------


async def test_compare_status_empty_hashes(lock_project: Path, mocker: MockerFixture) -> None:
    """Empty local hashes returns empty status."""

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    result = await transfer.compare_status(set(), mock_remote, mock_state, "origin")

    assert result == RemoteStatus(local_only=set(), remote_only=set(), common=set())
    mock_remote.bulk_exists.assert_not_called()


async def test_compare_status_all_known_in_index(lock_project: Path, mocker: MockerFixture) -> None:
    """All hashes known in index skips remote check."""

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    local_hashes = {"abc123def4567890", "def456abc7890123"}
    mock_state.remote_hashes_intersection.return_value = local_hashes

    result = await transfer.compare_status(local_hashes, mock_remote, mock_state, "origin")

    assert result["local_only"] == set()
    assert result["common"] == local_hashes
    mock_remote.bulk_exists.assert_not_called()


async def test_compare_status_queries_unknown(lock_project: Path, mocker: MockerFixture) -> None:
    """Unknown hashes query remote and update index."""

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    local_hashes = {"abc123def4567890", "def456abc7890123", "111222333444555a"}
    mock_state.remote_hashes_intersection.return_value = {"abc123def4567890"}
    mock_remote.bulk_exists = mocker.AsyncMock(
        return_value={"def456abc7890123": True, "111222333444555a": False}
    )

    result = await transfer.compare_status(local_hashes, mock_remote, mock_state, "origin")

    assert result["local_only"] == {"111222333444555a"}
    assert result["common"] == {"abc123def4567890", "def456abc7890123"}
    mock_state.remote_hashes_add.assert_called_once_with("origin", {"def456abc7890123"})


# -----------------------------------------------------------------------------
# Push Tests (test async functions directly to avoid nested event loops)
# -----------------------------------------------------------------------------


async def test_push_async_no_local_hashes(lock_project: Path, mocker: MockerFixture) -> None:
    """Push with no local hashes returns zero summary."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    result = await transfer._push_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    assert result["transferred"] == 0
    assert result["skipped"] == 0
    assert result["failed"] == 0


async def test_push_async_all_already_on_remote(lock_project: Path, mocker: MockerFixture) -> None:
    """Push when all files on remote returns skipped count."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_hashes_intersection.return_value = {hash1}

    result = await transfer._push_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    assert result["transferred"] == 0
    assert result["skipped"] == 1
    assert result["failed"] == 0


async def test_push_async_uploads_missing(lock_project: Path, mocker: MockerFixture) -> None:
    """Push uploads files not on remote."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    result = await transfer._push_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    assert result["transferred"] == 1
    assert result["skipped"] == 0
    assert result["failed"] == 0
    mock_state.remote_hashes_add.assert_called()


async def test_push_async_handles_failures(lock_project: Path, mocker: MockerFixture) -> None:
    """Push reports failures in summary."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=False, error="Upload failed")]
    )

    result = await transfer._push_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    assert result["transferred"] == 0
    assert result["failed"] == 1
    assert "Upload failed" in result["errors"]


async def test_push_async_with_stages(lock_project: Path, mocker: MockerFixture) -> None:
    """Push with specific stages only pushes those stage outputs."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    _helper_write_lock(state_dir, "my_stage", outs={None: FileHash(hash=hash1)})

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    result = await transfer._push_async(
        cache_dir,
        state_dir,
        mock_remote,
        mock_state,
        "origin",
        targets=["my_stage"],
        all_stages=_helper_make_all_stages("my_stage"),
    )

    assert result["transferred"] == 1


# -----------------------------------------------------------------------------
# Pull Tests (test async functions directly to avoid nested event loops)
# -----------------------------------------------------------------------------


async def test_pull_async_no_needed_hashes(lock_project: Path, mocker: MockerFixture) -> None:
    """Pull with no needed hashes returns zero summary."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    result = await transfer._pull_async(
        cache_dir, state_dir, mock_remote, mock_state, "origin", targets=["nonexistent"]
    )

    assert result["transferred"] == 0
    assert result["skipped"] == 0
    assert result["failed"] == 0


async def test_pull_async_all_already_local(lock_project: Path, mocker: MockerFixture) -> None:
    """Pull when all files local returns skipped count."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    _helper_write_lock(state_dir, "my_stage", outs={None: FileHash(hash=hash1)})

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    result = await transfer._pull_async(
        cache_dir,
        state_dir,
        mock_remote,
        mock_state,
        "origin",
        targets=["my_stage"],
        all_stages=_helper_make_all_stages("my_stage"),
    )

    assert result["transferred"] == 0
    assert result["skipped"] == 1
    assert result["failed"] == 0


async def test_pull_async_downloads_missing(lock_project: Path, mocker: MockerFixture) -> None:
    """Pull downloads files not in local cache."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"

    hash1 = "ab" + "c" * 14
    _helper_write_lock(state_dir, "my_stage", outs={None: FileHash(hash=hash1)})

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_remote.download_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    result = await transfer._pull_async(
        cache_dir,
        state_dir,
        mock_remote,
        mock_state,
        "origin",
        targets=["my_stage"],
        all_stages=_helper_make_all_stages("my_stage"),
    )

    assert result["transferred"] == 1
    assert result["skipped"] == 0
    assert result["failed"] == 0
    mock_state.remote_hashes_add.assert_called()


async def test_pull_async_without_stages_lists_remote(
    lock_project: Path, mocker: MockerFixture
) -> None:
    """Pull without stages lists all hashes from remote."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    (cache_dir / "files").mkdir(parents=True)

    hash1 = "ab" + "c" * 14
    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_remote.list_hashes = mocker.AsyncMock(return_value={hash1})
    mock_remote.download_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    result = await transfer._pull_async(
        cache_dir, state_dir, mock_remote, mock_state, "origin", targets=None
    )

    assert result["transferred"] == 1
    mock_remote.list_hashes.assert_called_once()


async def test_pull_async_handles_failures(lock_project: Path, mocker: MockerFixture) -> None:
    """Pull reports failures in summary."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"

    hash1 = "ab" + "c" * 14
    _helper_write_lock(state_dir, "my_stage", outs={None: FileHash(hash=hash1)})

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_remote.download_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=False, error="Download failed")]
    )

    result = await transfer._pull_async(
        cache_dir,
        state_dir,
        mock_remote,
        mock_state,
        "origin",
        targets=["my_stage"],
        all_stages=_helper_make_all_stages("my_stage"),
    )

    assert result["transferred"] == 0
    assert result["failed"] == 1
    assert "Download failed" in result["errors"]


async def test_pull_async_includes_deps(lock_project: Path, mocker: MockerFixture) -> None:
    """Pull includes the hashes of upstream stage outputs a target stage depends on."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"

    out_hash = "ab" + "c" * 14
    dep_hash = "de" + "f" * 14
    _helper_write_lock(
        state_dir,
        "my_stage",
        outs={None: FileHash(hash=out_hash)},
        deps={"upstream": FileHash(hash=dep_hash)},
    )

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_remote.download_batch = mocker.AsyncMock(
        return_value=[
            TransferResult(hash=out_hash, success=True),
            TransferResult(hash=dep_hash, success=True),
        ]
    )

    result = await transfer._pull_async(
        cache_dir,
        state_dir,
        mock_remote,
        mock_state,
        "origin",
        targets=["my_stage"],
        all_stages=_helper_make_all_stages("my_stage", "upstream"),
    )

    requested = {hash_ for hash_, _ in mock_remote.download_batch.call_args.args[0]}
    assert requested == {out_hash, dep_hash}, "Should request the output and the upstream dep"
    assert result["transferred"] == 2


# -----------------------------------------------------------------------------
# Utility Function Tests
# -----------------------------------------------------------------------------


def test_create_remote_from_name(lock_project: Path, mocker: MockerFixture) -> None:
    """Creates S3Remote from configured remote name."""

    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(remote_config, "get_default_remote", return_value="origin")

    remote, name = transfer.create_remote_from_name("origin")

    assert name == "origin"
    assert isinstance(remote, remote_mod.S3Remote)
    assert remote.bucket == "bucket"


def test_create_remote_from_name_default(lock_project: Path, mocker: MockerFixture) -> None:
    """Uses default remote when name is None."""

    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(remote_config, "get_default_remote", return_value="origin")

    remote, name = transfer.create_remote_from_name(None)

    assert name == "origin"
    assert isinstance(remote, remote_mod.S3Remote)
    assert remote.bucket == "bucket"


def test_create_remote_from_name_single_remote(lock_project: Path, mocker: MockerFixture) -> None:
    """Uses single remote when no default set."""

    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(remote_config, "get_default_remote", return_value=None)
    mocker.patch.object(remote_config, "list_remotes", return_value={"myremote": "s3://b/p"})

    remote, name = transfer.create_remote_from_name(None)

    assert name == "myremote"


def test_create_remote_from_name_multiple_remotes_error(
    lock_project: Path, mocker: MockerFixture
) -> None:
    """Raises error when multiple remotes and no default."""

    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(remote_config, "get_default_remote", return_value=None)
    mocker.patch.object(
        remote_config, "list_remotes", return_value={"r1": "s3://b1/p", "r2": "s3://b2/p"}
    )

    with pytest.raises(exceptions.RemoteNotFoundError, match="Could not determine remote name"):
        transfer.create_remote_from_name(None)


# ============================================================================
# Integration Tests (with moto)
# ============================================================================


async def test_push_async_integration(
    tmp_path: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """Integration: push files to real moto S3."""
    state_dir = tmp_path / ".pivot"
    cache_dir = state_dir / "cache"
    files_dir = cache_dir / "files"
    files_dir.mkdir(parents=True)

    hash1 = "ab" + "c" * 14
    hash2 = "de" + "f" * 14

    cache_path1 = cache_mod.get_cache_path(files_dir, hash1)
    cache_path1.parent.mkdir(parents=True, exist_ok=True)
    cache_path1.write_bytes(b"content1")

    cache_path2 = cache_mod.get_cache_path(files_dir, hash2)
    cache_path2.parent.mkdir(parents=True, exist_ok=True)
    cache_path2.write_bytes(b"content2")

    state_dir.mkdir(parents=True, exist_ok=True)
    state_db = state_mod.StateDB(state_dir)

    await transfer._push_async(cache_dir, state_dir, s3_remote, state_db, "origin")

    response = await aioboto3_s3_client.get_object(
        Bucket=s3_remote.bucket,
        Key=keys.hash_to_key(s3_remote.prefix, hash1),
    )
    content = await response["Body"].read()
    assert content == b"content1"
    state_db.close()


async def test_push_async_directory_target_sends_listing_and_files_integration(
    lock_project: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """A targeted directory push sends its listing object and every listed file object."""
    state_dir = lock_project / ".pivot"
    cache_dir = state_dir / "cache"
    files_dir = cache_dir / "files"
    first = b"first file"
    second = b"second file"
    first_hash = _helper_cache_object(files_dir, first)
    second_hash = _helper_cache_object(files_dir, second)
    directory = _helper_hashed_dir(
        ("a.txt", first_hash, len(first)),
        ("nested/b.txt", second_hash, len(second)),
    )
    _helper_write_lock(
        state_dir,
        "my_stage",
        outs={"output_dir": directory["record"]},
        directories=(directory,),
    )
    assert cache_mod.cached_listing(files_dir, directory["record"]) is None

    with state_mod.StateDB(state_dir) as state_db:
        result = await transfer._push_async(
            cache_dir,
            state_dir,
            s3_remote,
            state_db,
            "origin",
            targets=["my_stage"],
            all_stages=_helper_make_all_stages("my_stage"),
        )

    assert result == {"transferred": 3, "skipped": 0, "failed": 0, "errors": []}
    assert (
        await _helper_remote_object(s3_remote, aioboto3_s3_client, directory["record"]["hash"])
        == directory["data"]
    )
    assert await _helper_remote_object(s3_remote, aioboto3_s3_client, first_hash) == first
    assert await _helper_remote_object(s3_remote, aioboto3_s3_client, second_hash) == second


async def test_pull_async_integration(
    tmp_path: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """Integration: pull files from real moto S3."""
    hash1 = xxhash.xxh64(b"content1").hexdigest()

    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=keys.hash_to_key(s3_remote.prefix, hash1),
        Body=b"content1",
    )

    state_dir = tmp_path / ".pivot"
    cache_dir = state_dir / "cache"
    cache_dir.mkdir(parents=True)

    state_db = state_mod.StateDB(state_dir)

    await transfer._pull_async(cache_dir, state_dir, s3_remote, state_db, "origin")

    files_dir = cache_dir / "files"
    cache_path = cache_mod.get_cache_path(files_dir, hash1)
    assert cache_path.read_bytes() == b"content1"
    state_db.close()


async def test_pull_then_checkout_preserves_executable_bit_integration(
    tmp_path: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """A directory output's executable file keeps its executable bit after `pivot pull`.

    A pull writes each downloaded cache object by content hash alone, with no
    executable bit (always 0o444). The pulled listing records the bit. Checking it out as
    HARDLINK or SYMLINK must still produce a correctly-executable workspace file, by falling
    back to COPY rather than serving the pulled object's own (wrong) permissions.
    """
    content = b"#!/bin/sh\necho hi\n"
    file_hash = xxhash.xxh64(content).hexdigest()
    members = listing.DirListing(
        files=[DirManifestEntry(relpath="run.sh", hash=file_hash, size=len(content), isexec=True)],
        empty_dirs=[],
    )
    listing_data = listing.encode(members)
    output_hash = listing.record_of(members, listing_data)

    await _helper_put_remote_object(s3_remote, aioboto3_s3_client, file_hash, content)
    await _helper_put_remote_object(
        s3_remote, aioboto3_s3_client, output_hash["hash"], listing_data
    )

    state_dir = tmp_path / ".pivot"
    cache_dir = state_dir / "cache"
    cache_dir.mkdir(parents=True)
    state_db = state_mod.StateDB(state_dir)

    await transfer._pull_async(cache_dir, state_dir, s3_remote, state_db, "origin")
    state_db.close()

    pulled_cache_path = cache_mod.get_cache_path(cache_dir / "files", file_hash)
    assert not os.access(pulled_cache_path, os.X_OK), "pull has no isexec to write; sanity check"
    assert cache_mod.cached_listing(cache_dir / "files", output_hash) == members

    output_dir = tmp_path / "restored"
    assert cache_mod.restore_from_cache(
        output_dir,
        output_hash,
        cache_dir / "files",
        checkout_modes=cache_mod.DEFAULT_CHECKOUT_MODE_ORDER,
    )

    assert os.access(output_dir / "run.sh", os.X_OK), "checkout must restore the executable bit"


async def test_push_pull_roundtrip_integration(
    tmp_path: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """Integration: push then pull to verify roundtrip."""
    hash1 = xxhash.xxh64(b"original").hexdigest()

    state_dir1 = tmp_path / ".pivot1"
    cache_dir1 = state_dir1 / "cache"
    files_dir1 = cache_dir1 / "files"
    files_dir1.mkdir(parents=True)

    cache_path1 = cache_mod.get_cache_path(files_dir1, hash1)
    cache_path1.parent.mkdir(parents=True, exist_ok=True)
    cache_path1.write_bytes(b"original")

    state_db1 = state_mod.StateDB(state_dir1)

    await transfer._push_async(cache_dir1, state_dir1, s3_remote, state_db1, "origin")

    state_dir2 = tmp_path / ".pivot2"
    cache_dir2 = state_dir2 / "cache"
    cache_dir2.mkdir(parents=True)
    state_db2 = state_mod.StateDB(state_dir2)

    await transfer._pull_async(cache_dir2, state_dir2, s3_remote, state_db2, "origin")

    files_dir2 = cache_dir2 / "files"
    cache_path2 = cache_mod.get_cache_path(files_dir2, hash1)
    assert cache_path2.read_bytes() == b"original"
    state_db1.close()
    state_db2.close()


async def test_compare_status_integration(
    tmp_path: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """Integration: compare local vs remote status.

    Note: compare_status only checks local hashes against remote, never
    enumerates remote-only files. So we only set up the common hash.
    """
    hash_common = "de" + "f" * 14
    hash_local_only = "12" + "3" * 14

    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=keys.hash_to_key(s3_remote.prefix, hash_common),
        Body=b"remote2",
    )

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_db = state_mod.StateDB(state_dir)

    status = await transfer.compare_status(
        {hash_common, hash_local_only},
        s3_remote,
        state_db,
        "origin",
    )

    assert status["remote_only"] == set()
    assert status["local_only"] == {hash_local_only}
    assert status["common"] == {hash_common}
    state_db.close()


async def test_pull_async_with_deps_integration(
    tmp_path: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """Integration: pull with dependency hashes."""
    out_hash = xxhash.xxh64(b"main_file").hexdigest()
    dep_hash = xxhash.xxh64(b"dependency_file").hexdigest()

    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=keys.hash_to_key(s3_remote.prefix, out_hash),
        Body=b"main_file",
    )
    await aioboto3_s3_client.put_object(
        Bucket=s3_remote.bucket,
        Key=keys.hash_to_key(s3_remote.prefix, dep_hash),
        Body=b"dependency_file",
    )

    state_dir = tmp_path / ".pivot"
    cache_dir = state_dir / "cache"
    cache_dir.mkdir(parents=True)

    _helper_write_lock(
        state_dir,
        "my_stage",
        outs={None: FileHash(hash=out_hash)},
        deps={"upstream": FileHash(hash=dep_hash)},
    )

    state_db = state_mod.StateDB(state_dir)

    await transfer._pull_async(
        cache_dir,
        state_dir,
        s3_remote,
        state_db,
        "origin",
        targets=["my_stage"],
        all_stages=_helper_make_all_stages("my_stage", "upstream"),
    )

    files_dir = cache_dir / "files"
    out_path = cache_mod.get_cache_path(files_dir, out_hash)
    dep_path = cache_mod.get_cache_path(files_dir, dep_hash)
    assert out_path.read_bytes() == b"main_file"
    assert dep_path.read_bytes() == b"dependency_file"
    state_db.close()


async def test_pull_async_directory_target_fetches_listing_then_files_integration(
    lock_project: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """A targeted directory whose listing is not local fetches its listing, then its files."""
    first = b"first pulled file"
    second = b"second pulled file"
    first_hash = xxhash.xxh64(first).hexdigest()
    second_hash = xxhash.xxh64(second).hexdigest()
    directory = _helper_hashed_dir(
        ("a.txt", first_hash, len(first)),
        ("nested/b.txt", second_hash, len(second)),
    )
    for object_hash, content in (
        (directory["record"]["hash"], directory["data"]),
        (first_hash, first),
        (second_hash, second),
    ):
        await _helper_put_remote_object(s3_remote, aioboto3_s3_client, object_hash, content)

    state_dir = lock_project / ".pivot"
    cache_dir = state_dir / "cache"
    files_dir = cache_dir / "files"
    stage_lock = _helper_write_lock(
        state_dir,
        "my_stage",
        outs={"output_dir": directory["record"]},
        directories=(directory,),
    )
    stage_lock.output_listing_path(ArtifactIdentity("my_stage", "output_dir")).unlink()

    with state_mod.StateDB(state_dir) as state_db:
        result = await transfer._pull_async(
            cache_dir,
            state_dir,
            s3_remote,
            state_db,
            "origin",
            targets=["my_stage"],
            all_stages=_helper_make_all_stages("my_stage"),
        )

    assert result == {"transferred": 3, "skipped": 0, "failed": 0, "errors": []}
    assert cache_mod.cached_listing(files_dir, directory["record"]) == directory["listing"]
    assert cache_mod.get_cache_path(files_dir, first_hash).read_bytes() == first
    assert cache_mod.get_cache_path(files_dir, second_hash).read_bytes() == second


async def test_pull_async_with_directory_dep_integration(
    lock_project: Path,
    s3_remote: remote_mod.S3Remote,
    aioboto3_s3_client: S3Client,
) -> None:
    """A targeted pull includes an upstream directory's listing object and listed files.

    The committed listing is copied into the local cache, so only file objects are fetched.
    """
    output = b"main output"
    dep_file = b"directory dependency file"
    out_hash = xxhash.xxh64(output).hexdigest()
    dep_file_hash = xxhash.xxh64(dep_file).hexdigest()
    directory = _helper_hashed_dir(("input.txt", dep_file_hash, len(dep_file)))
    for object_hash, content in ((out_hash, output), (dep_file_hash, dep_file)):
        await _helper_put_remote_object(s3_remote, aioboto3_s3_client, object_hash, content)

    state_dir = lock_project / ".pivot"
    cache_dir = state_dir / "cache"
    files_dir = cache_dir / "files"
    _helper_write_lock(
        state_dir,
        "upstream",
        outs={None: directory["record"]},
        directories=(directory,),
    )
    _helper_write_lock(
        state_dir,
        "my_stage",
        outs={None: FileHash(hash=out_hash)},
        deps={"upstream": directory["record"]},
    )

    with state_mod.StateDB(state_dir) as state_db:
        result = await transfer._pull_async(
            cache_dir,
            state_dir,
            s3_remote,
            state_db,
            "origin",
            targets=["my_stage"],
            all_stages=_helper_make_all_stages("my_stage", "upstream"),
        )

    assert result == {"transferred": 2, "skipped": 1, "failed": 0, "errors": []}
    assert cache_mod.get_cache_path(files_dir, out_hash).read_bytes() == output
    assert cache_mod.cached_listing(files_dir, directory["record"]) == directory["listing"]
    assert cache_mod.get_cache_path(files_dir, dep_file_hash).read_bytes() == dep_file


# -----------------------------------------------------------------------------
# URL Change Detection Tests
# -----------------------------------------------------------------------------


async def test_push_url_change_clears_index(lock_project: Path, mocker: MockerFixture) -> None:
    """Push detects URL change and clears index."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_remote.url = "s3://bucket/new-prefix"
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_get_url.return_value = "s3://bucket/old-prefix"
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    await transfer._push_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    mock_state.remote_index_clear.assert_called_once_with("origin")
    mock_state.remote_set_url.assert_called_once_with("origin", "s3://bucket/new-prefix")


async def test_push_first_use_sets_url_without_clear(
    lock_project: Path, mocker: MockerFixture
) -> None:
    """Push on first use sets URL without clearing."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_remote.url = "s3://bucket/prefix"
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_get_url.return_value = None
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    await transfer._push_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    mock_state.remote_set_url.assert_called_once_with("origin", "s3://bucket/prefix")
    mock_state.remote_index_clear.assert_not_called()


async def test_push_same_url_no_clear(lock_project: Path, mocker: MockerFixture) -> None:
    """Push with same URL does not clear index."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_remote.url = "s3://bucket/prefix"
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_get_url.return_value = "s3://bucket/prefix"
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    await transfer._push_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    mock_state.remote_index_clear.assert_not_called()
    mock_state.remote_set_url.assert_not_called()


async def test_pull_url_change_clears_index(lock_project: Path, mocker: MockerFixture) -> None:
    """Pull detects URL change and clears index."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    (cache_dir / "files").mkdir(parents=True)

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_remote.url = "s3://bucket/new-prefix"
    mock_remote.list_hashes = mocker.AsyncMock(return_value=set())
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_get_url.return_value = "s3://bucket/old-prefix"

    await transfer._pull_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    mock_state.remote_index_clear.assert_called_once_with("origin")
    mock_state.remote_set_url.assert_called_once_with("origin", "s3://bucket/new-prefix")


async def test_pull_first_use_sets_url_without_clear(
    lock_project: Path, mocker: MockerFixture
) -> None:
    """Pull on first use sets URL without clearing."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    (cache_dir / "files").mkdir(parents=True)

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_remote.url = "s3://bucket/prefix"
    mock_remote.list_hashes = mocker.AsyncMock(return_value=set())
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_get_url.return_value = None

    await transfer._pull_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    mock_state.remote_set_url.assert_called_once_with("origin", "s3://bucket/prefix")
    mock_state.remote_index_clear.assert_not_called()


async def test_pull_same_url_no_clear(lock_project: Path, mocker: MockerFixture) -> None:
    """Pull with same URL does not clear index."""

    cache_dir = lock_project / ".pivot" / "cache"
    state_dir = lock_project / ".pivot"
    (cache_dir / "files").mkdir(parents=True)

    mock_remote = mocker.Mock(spec=remote_mod.S3Remote)
    mock_remote.url = "s3://bucket/prefix"
    mock_remote.list_hashes = mocker.AsyncMock(return_value=set())
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_get_url.return_value = "s3://bucket/prefix"

    await transfer._pull_async(cache_dir, state_dir, mock_remote, mock_state, "origin")

    mock_state.remote_index_clear.assert_not_called()
    mock_state.remote_set_url.assert_not_called()
