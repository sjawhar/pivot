from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import transfer

if TYPE_CHECKING:
    import pathlib

    import pytest_mock

# -----------------------------------------------------------------------------
# Local Cache Hash Scanning Tests
# -----------------------------------------------------------------------------


def test_get_local_cache_hashes_empty(tmp_path: pathlib.Path) -> None:
    """Empty cache returns empty set."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    result = transfer.get_local_cache_hashes(cache_dir)
    assert result == set()


def test_get_local_cache_hashes_no_files_dir(tmp_path: pathlib.Path) -> None:
    """Missing files directory returns empty set."""
    cache_dir = tmp_path / "cache"

    result = transfer.get_local_cache_hashes(cache_dir)
    assert result == set()


def test_get_local_cache_hashes(tmp_path: pathlib.Path) -> None:
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


def test_get_local_cache_hashes_ignores_invalid_structure(tmp_path: pathlib.Path) -> None:
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
# Stage Output Hash Extraction Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def lock_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    from pivot import project

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot" / "cache" / "stages").mkdir(parents=True)
    return tmp_path


def test_get_stage_output_hashes_no_lock(lock_project: pathlib.Path) -> None:
    """Missing lock file returns empty set with warning."""
    cache_dir = lock_project / ".pivot" / "cache"

    result = transfer.get_stage_output_hashes(cache_dir, ["nonexistent"])
    assert result == set()


def test_get_stage_output_hashes_file_output(lock_project: pathlib.Path) -> None:
    """Extracts hash from file output in lock file."""
    cache_dir = lock_project / ".pivot" / "cache"

    lock_data = {
        "outs": [{"path": "output.csv", "hash": "abc123def45678"}],
    }
    lock_path = cache_dir / "stages" / "my_stage.lock"
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    result = transfer.get_stage_output_hashes(cache_dir, ["my_stage"])
    assert result == {"abc123def45678"}


def test_get_stage_output_hashes_directory_output(lock_project: pathlib.Path) -> None:
    """Extracts all hashes from directory output including manifest."""
    cache_dir = lock_project / ".pivot" / "cache"

    lock_data = {
        "outs": [
            {
                "path": "output_dir",
                "hash": "treehash1234567",
                "manifest": [
                    {"relpath": "file1.txt", "hash": "filehash1234567", "size": 100},
                    {"relpath": "file2.txt", "hash": "filehash2345678", "size": 200},
                ],
            }
        ],
    }
    lock_path = cache_dir / "stages" / "my_stage.lock"
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    result = transfer.get_stage_output_hashes(cache_dir, ["my_stage"])
    assert result == {"treehash1234567", "filehash1234567", "filehash2345678"}


def test_get_stage_output_hashes_multiple_stages(lock_project: pathlib.Path) -> None:
    """Collects hashes from multiple stages."""
    cache_dir = lock_project / ".pivot" / "cache"

    for i, stage in enumerate(["stage_a", "stage_b"]):
        lock_data = {"outs": [{"path": f"out{i}.csv", "hash": f"hash{i}{'0' * 11}"}]}
        lock_path = cache_dir / "stages" / f"{stage}.lock"
        with lock_path.open("w") as f:
            yaml.dump(lock_data, f)

    result = transfer.get_stage_output_hashes(cache_dir, ["stage_a", "stage_b"])
    assert "hash0" + "0" * 11 in result
    assert "hash1" + "0" * 11 in result


def test_get_stage_output_hashes_skips_uncached(lock_project: pathlib.Path) -> None:
    """Skips outputs with null hash (uncached)."""
    cache_dir = lock_project / ".pivot" / "cache"

    lock_data = {
        "outs": [
            {"path": "cached.csv", "hash": "abc123def45678"},
            {"path": "uncached.csv", "hash": None},
        ],
    }
    lock_path = cache_dir / "stages" / "my_stage.lock"
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    result = transfer.get_stage_output_hashes(cache_dir, ["my_stage"])
    assert result == {"abc123def45678"}


# -----------------------------------------------------------------------------
# Stage Dependency Hash Extraction Tests
# -----------------------------------------------------------------------------


def test_get_stage_dep_hashes(lock_project: pathlib.Path) -> None:
    """Extracts dependency hashes from lock file."""
    cache_dir = lock_project / ".pivot" / "cache"

    lock_data = {
        "deps": [
            {"path": "input.csv", "hash": "dep1hash1234567"},
            {"path": "config.yaml", "hash": "dep2hash1234567"},
        ],
    }
    lock_path = cache_dir / "stages" / "my_stage.lock"
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    result = transfer.get_stage_dep_hashes(cache_dir, ["my_stage"])
    assert result == {"dep1hash1234567", "dep2hash1234567"}


def test_get_stage_dep_hashes_with_manifest(lock_project: pathlib.Path) -> None:
    """Extracts all hashes from directory dependency including manifest."""
    cache_dir = lock_project / ".pivot" / "cache"

    lock_data = {
        "deps": [
            {
                "path": "input_dir",
                "hash": "dirtreehash1234",
                "manifest": [
                    {"relpath": "a.txt", "hash": "afilehash123456", "size": 10},
                    {"relpath": "b.txt", "hash": "bfilehash123456", "size": 20},
                ],
            }
        ],
    }
    lock_path = cache_dir / "stages" / "my_stage.lock"
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    result = transfer.get_stage_dep_hashes(cache_dir, ["my_stage"])
    assert result == {"dirtreehash1234", "afilehash123456", "bfilehash123456"}


def test_get_stage_dep_hashes_no_lock(lock_project: pathlib.Path) -> None:
    """Missing lock file skips silently."""
    cache_dir = lock_project / ".pivot" / "cache"

    result = transfer.get_stage_dep_hashes(cache_dir, ["nonexistent"])
    assert result == set()


# -----------------------------------------------------------------------------
# Compare Status Tests
# -----------------------------------------------------------------------------


async def test_compare_status_empty_hashes(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Empty local hashes returns empty status."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote
    from pivot.types import RemoteStatus

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    result = await transfer.compare_status(set(), mock_remote, mock_state, "origin")

    assert result == RemoteStatus(local_only=set(), remote_only=set(), common=set())
    mock_remote.bulk_exists.assert_not_called()


async def test_compare_status_all_known_in_index(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """All hashes known in index skips remote check."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    local_hashes = {"abc123def4567890", "def456abc7890123"}
    mock_state.remote_hashes_intersection.return_value = local_hashes

    result = await transfer.compare_status(local_hashes, mock_remote, mock_state, "origin")

    assert result["local_only"] == set()
    assert result["common"] == local_hashes
    mock_remote.bulk_exists.assert_not_called()


async def test_compare_status_queries_unknown(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Unknown hashes query remote and update index."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote

    mock_remote = mocker.Mock(spec=S3Remote)
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


async def test_push_async_no_local_hashes(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Push with no local hashes returns zero summary."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote

    cache_dir = lock_project / ".pivot" / "cache"
    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    result = await transfer._push_async(cache_dir, mock_remote, mock_state, "origin")

    assert result["transferred"] == 0
    assert result["skipped"] == 0
    assert result["failed"] == 0


async def test_push_async_all_already_on_remote(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Push when all files on remote returns skipped count."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote

    cache_dir = lock_project / ".pivot" / "cache"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_hashes_intersection.return_value = {hash1}

    result = await transfer._push_async(cache_dir, mock_remote, mock_state, "origin")

    assert result["transferred"] == 0
    assert result["skipped"] == 1
    assert result["failed"] == 0


async def test_push_async_uploads_missing(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Push uploads files not on remote."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote
    from pivot.types import TransferResult

    cache_dir = lock_project / ".pivot" / "cache"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    result = await transfer._push_async(cache_dir, mock_remote, mock_state, "origin")

    assert result["transferred"] == 1
    assert result["skipped"] == 0
    assert result["failed"] == 0
    mock_state.remote_hashes_add.assert_called()


async def test_push_async_handles_failures(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Push reports failures in summary."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote
    from pivot.types import TransferResult

    cache_dir = lock_project / ".pivot" / "cache"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=False, error="Upload failed")]
    )

    result = await transfer._push_async(cache_dir, mock_remote, mock_state, "origin")

    assert result["transferred"] == 0
    assert result["failed"] == 1
    assert "Upload failed" in result["errors"]


async def test_push_async_with_stages(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Push with specific stages only pushes those stage outputs."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote
    from pivot.types import TransferResult

    cache_dir = lock_project / ".pivot" / "cache"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    lock_data = {"outs": [{"path": "out.csv", "hash": hash1}]}
    lock_path = cache_dir / "stages" / "my_stage.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_state.remote_hashes_intersection.return_value = set()
    mock_remote.bulk_exists = mocker.AsyncMock(return_value={hash1: False})
    mock_remote.upload_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    result = await transfer._push_async(
        cache_dir, mock_remote, mock_state, "origin", targets=["my_stage"]
    )

    assert result["transferred"] == 1


# -----------------------------------------------------------------------------
# Pull Tests (test async functions directly to avoid nested event loops)
# -----------------------------------------------------------------------------


async def test_pull_async_no_needed_hashes(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Pull with no needed hashes returns zero summary."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote

    cache_dir = lock_project / ".pivot" / "cache"
    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    result = await transfer._pull_async(
        cache_dir, mock_remote, mock_state, "origin", targets=["nonexistent"]
    )

    assert result["transferred"] == 0
    assert result["skipped"] == 0
    assert result["failed"] == 0


async def test_pull_async_all_already_local(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Pull when all files local returns skipped count."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote

    cache_dir = lock_project / ".pivot" / "cache"
    files_dir = cache_dir / "files"

    hash1 = "ab" + "c" * 14
    (files_dir / "ab").mkdir(parents=True)
    (files_dir / "ab" / ("c" * 14)).write_text("content1")

    lock_data = {"outs": [{"path": "out.csv", "hash": hash1}]}
    lock_path = cache_dir / "stages" / "my_stage.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)

    result = await transfer._pull_async(
        cache_dir, mock_remote, mock_state, "origin", targets=["my_stage"]
    )

    assert result["transferred"] == 0
    assert result["skipped"] == 1
    assert result["failed"] == 0


async def test_pull_async_downloads_missing(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Pull downloads files not in local cache."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote
    from pivot.types import TransferResult

    cache_dir = lock_project / ".pivot" / "cache"

    hash1 = "ab" + "c" * 14
    lock_data = {"outs": [{"path": "out.csv", "hash": hash1}]}
    lock_path = cache_dir / "stages" / "my_stage.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_remote.download_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    result = await transfer._pull_async(
        cache_dir, mock_remote, mock_state, "origin", targets=["my_stage"]
    )

    assert result["transferred"] == 1
    assert result["skipped"] == 0
    assert result["failed"] == 0
    mock_state.remote_hashes_add.assert_called()


async def test_pull_async_without_stages_lists_remote(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Pull without stages lists all hashes from remote."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote
    from pivot.types import TransferResult

    cache_dir = lock_project / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)

    hash1 = "ab" + "c" * 14
    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_remote.list_hashes = mocker.AsyncMock(return_value={hash1})
    mock_remote.download_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=True)]
    )

    result = await transfer._pull_async(cache_dir, mock_remote, mock_state, "origin", targets=None)

    assert result["transferred"] == 1
    mock_remote.list_hashes.assert_called_once()


async def test_pull_async_handles_failures(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Pull reports failures in summary."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote
    from pivot.types import TransferResult

    cache_dir = lock_project / ".pivot" / "cache"

    hash1 = "ab" + "c" * 14
    lock_data = {"outs": [{"path": "out.csv", "hash": hash1}]}
    lock_path = cache_dir / "stages" / "my_stage.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_remote.download_batch = mocker.AsyncMock(
        return_value=[TransferResult(hash=hash1, success=False, error="Download failed")]
    )

    result = await transfer._pull_async(
        cache_dir, mock_remote, mock_state, "origin", targets=["my_stage"]
    )

    assert result["transferred"] == 0
    assert result["failed"] == 1
    assert "Download failed" in result["errors"]


async def test_pull_async_includes_deps(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Pull includes dependency hashes when stages specified."""
    from pivot import state as state_mod
    from pivot.remote import S3Remote
    from pivot.types import TransferResult

    cache_dir = lock_project / ".pivot" / "cache"

    out_hash = "ab" + "c" * 14
    dep_hash = "de" + "f" * 14
    lock_data = {
        "outs": [{"path": "out.csv", "hash": out_hash}],
        "deps": [{"path": "in.csv", "hash": dep_hash}],
    }
    lock_path = cache_dir / "stages" / "my_stage.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as f:
        yaml.dump(lock_data, f)

    mock_remote = mocker.Mock(spec=S3Remote)
    mock_state = mocker.Mock(spec=state_mod.StateDB)
    mock_remote.download_batch = mocker.AsyncMock(
        return_value=[
            TransferResult(hash=out_hash, success=True),
            TransferResult(hash=dep_hash, success=True),
        ]
    )

    result = await transfer._pull_async(
        cache_dir, mock_remote, mock_state, "origin", targets=["my_stage"]
    )

    assert result["transferred"] == 2


# -----------------------------------------------------------------------------
# Utility Function Tests
# -----------------------------------------------------------------------------


def test_get_default_cache_dir(lock_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns cache dir relative to project root."""
    result = transfer.get_default_cache_dir()
    assert result == lock_project / ".pivot" / "cache"


def test_create_remote_from_name(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Creates S3Remote from configured remote name."""
    from pivot import remote_config

    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(remote_config, "get_default_remote", return_value="origin")

    remote, name = transfer.create_remote_from_name("origin")

    assert name == "origin"
    assert remote.bucket == "bucket"


def test_create_remote_from_name_default(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Uses default remote when name is None."""
    from pivot import remote_config

    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(remote_config, "get_default_remote", return_value="origin")

    remote, name = transfer.create_remote_from_name(None)

    assert name == "origin"
    assert remote.bucket == "bucket"


def test_create_remote_from_name_single_remote(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Uses single remote when no default set."""
    from pivot import remote_config

    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(remote_config, "get_default_remote", return_value=None)
    mocker.patch.object(remote_config, "list_remotes", return_value={"myremote": "s3://b/p"})

    remote, name = transfer.create_remote_from_name(None)

    assert name == "myremote"


def test_create_remote_from_name_multiple_remotes_error(
    lock_project: pathlib.Path, mocker: pytest_mock.MockerFixture
) -> None:
    """Raises error when multiple remotes and no default."""
    from pivot import exceptions, remote_config

    mocker.patch.object(remote_config, "get_remote_url", return_value="s3://bucket/prefix")
    mocker.patch.object(remote_config, "get_default_remote", return_value=None)
    mocker.patch.object(
        remote_config, "list_remotes", return_value={"r1": "s3://b1/p", "r2": "s3://b2/p"}
    )

    with pytest.raises(exceptions.RemoteNotFoundError, match="Could not determine remote name"):
        transfer.create_remote_from_name(None)
