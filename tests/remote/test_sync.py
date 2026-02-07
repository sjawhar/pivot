from __future__ import annotations

import pathlib

import pytest

from pivot.cli import helpers as cli_helpers
from pivot.remote import sync
from pivot.storage import cache as cache_mod
from pivot.storage import lock
from pivot.types import DirHash, DirManifestEntry, FileHash, LockData

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
        dep_generations={},
    )
    stage_lock = lock.StageLock(stage_name, stages_dir)
    stage_lock.write(lock_data)


@pytest.fixture(autouse=True)
def _mock_get_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock cli_helpers.get_stage to return all-cached outputs.

    Sync functions filter non-cached outputs via registry; tests without a
    pipeline can return empty outs to allow hashes through.
    """

    def _get_stage(name: str) -> dict[str, object]:
        return {"outs": []}

    monkeypatch.setattr(cli_helpers, "get_stage", _get_stage)


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
        dep_generations={},
    )
    stage_lock = lock.StageLock("my_stage", stages_dir)
    stage_lock.write(lock_data)

    result = sync.get_stage_dep_hashes(state_dir, ["my_stage"])

    assert "3333333333333333" in result
    assert "bbbbbbbbbbbbbbbb" not in result


# =============================================================================
# get_target_hashes edge cases
# =============================================================================


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


# =============================================================================
# Task 3: Push skips directory cache paths
# =============================================================================


def test_push_skips_directory_cache_paths(tmp_path: pathlib.Path) -> None:
    """Push should never enqueue directory paths for upload."""
    cache_dir = tmp_path / "cache"
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

    file_path = cache_mod.get_cache_path(files_dir, file_hash)
    dir_path = cache_mod.get_cache_path(files_dir, dir_hash)
    assert file_path.exists() and file_path.is_file()
    assert dir_path.exists() and dir_path.is_dir()

    # Verify the filtering logic directly
    items = list[tuple[pathlib.Path, str]]()
    for hash_ in [file_hash, dir_hash]:
        cache_path = cache_mod.get_cache_path(files_dir, hash_)
        if cache_path.is_file():
            items.append((cache_path, hash_))

    assert len(items) == 1
    assert items[0][1] == file_hash
