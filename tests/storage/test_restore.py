from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import exceptions, project
from pivot.storage import cache, restore

if TYPE_CHECKING:
    from conftest import GitRepo


# =============================================================================
# _parse_lock_data_from_bytes Tests
# =============================================================================


def test_parse_lock_data_from_bytes_valid() -> None:
    """Parses valid lock file content."""
    content = b"""
code_manifest:
  func:main: abc123
params:
  lr: 0.01
deps:
  - path: data.csv
    hash: def456
outs:
  - path: model.pkl
    hash: ghi789
dep_generations: {}
"""
    result = restore._parse_lock_data_from_bytes(content)

    assert result is not None
    assert "outs" in result
    assert result["outs"][0]["path"] == "model.pkl"


def test_parse_lock_data_from_bytes_invalid() -> None:
    """Returns None for invalid YAML or YAML missing 'outs' key."""
    content = b"not: valid: yaml: content"

    result = restore._parse_lock_data_from_bytes(content)

    # Either invalid YAML returns None, or valid YAML without 'outs' returns None
    assert result is None


def test_parse_lock_data_from_bytes_missing_outs() -> None:
    """Returns None when outs key is missing."""
    content = b"""
code_manifest:
  func:main: abc123
"""
    result = restore._parse_lock_data_from_bytes(content)

    assert result is None


# =============================================================================
# _parse_pvt_data_from_bytes Tests
# =============================================================================


def test_parse_pvt_data_from_bytes_valid() -> None:
    """Parses valid pvt file content."""
    content = b"""
path: data.csv
hash: abc123
size: 1024
"""
    result = restore._parse_pvt_data_from_bytes(content)

    assert result is not None
    assert result["path"] == "data.csv"
    assert result["hash"] == "abc123"


def test_parse_pvt_data_from_bytes_path_traversal() -> None:
    """Returns None for paths with traversal."""
    content = b"""
path: ../../../etc/passwd
hash: abc123
size: 100
"""
    result = restore._parse_pvt_data_from_bytes(content)

    assert result is None


def test_parse_pvt_data_from_bytes_missing_keys() -> None:
    """Returns None when required keys are missing."""
    content = b"""
path: data.csv
"""
    result = restore._parse_pvt_data_from_bytes(content)

    assert result is None


# =============================================================================
# resolve_targets Tests
# =============================================================================


def test_resolve_targets_as_stage(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves target as stage name when lock file exists."""
    repo_path, commit = git_repo

    # Create lock file for stage
    cache_dir = repo_path / ".pivot" / "cache"
    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "code_manifest": {"func:main": "abc123"},
        "params": {},
        "deps": [],
        "outs": [{"path": "model.pkl", "hash": "def456"}],
        "dep_generations": {},
    }
    (repo_path / ".pivot" / "stages" / "train.lock").write_text(yaml.dump(lock_content))

    sha = commit("add lock file")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    targets = restore.resolve_targets(["train"], sha, cache_dir)

    assert len(targets) == 1
    assert targets[0]["target_type"] == "stage"
    assert targets[0]["original_target"] == "train"
    assert "model.pkl" in targets[0]["paths"]


def test_resolve_targets_as_pvt_file(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves target as pvt tracked file."""
    repo_path, commit = git_repo

    # Create pvt file
    pvt_content = {"path": "data.csv", "hash": "abc123", "size": 1024}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"
    targets = restore.resolve_targets(["data.csv"], sha, cache_dir)

    assert len(targets) == 1
    assert targets[0]["target_type"] == "file"
    assert targets[0]["hashes"]["data.csv"] is not None
    assert targets[0]["hashes"]["data.csv"]["hash"] == "abc123"  # type: ignore[index]


def test_resolve_targets_as_git_file(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves target as plain git-tracked file."""
    repo_path, commit = git_repo

    # Create and commit a regular file
    (repo_path / "readme.txt").write_text("readme content")
    sha = commit("add readme")[:7]

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"
    targets = restore.resolve_targets(["readme.txt"], sha, cache_dir)

    assert len(targets) == 1
    assert targets[0]["target_type"] == "file"
    assert targets[0]["hashes"]["readme.txt"] is None  # No hash for git-only files


def test_resolve_targets_not_found(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises TargetNotFoundError for unknown tarrestore."""
    repo_path, commit = git_repo

    (repo_path / "dummy.txt").write_text("dummy")
    sha = commit("initial commit")[:7]

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    with pytest.raises(exceptions.TargetNotFoundError, match="nonexistent"):
        restore.resolve_targets(["nonexistent"], sha, cache_dir)


def test_resolve_targets_path_traversal(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises TargetNotFoundError for path traversal in tarrestore."""
    repo_path, commit = git_repo

    (repo_path / "dummy.txt").write_text("dummy")
    sha = commit("initial commit")[:7]

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    with pytest.raises(exceptions.TargetNotFoundError, match="Path traversal"):
        restore.resolve_targets(["../../../etc/passwd"], sha, cache_dir)


# =============================================================================
# restore_targets_from_revision Tests
# =============================================================================


def test_restore_targets_from_revision_invalid_rev(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raises RevisionNotFoundError for invalid revision."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    commit("initial")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    with pytest.raises(exceptions.RevisionNotFoundError, match="invalid-rev"):
        restore.restore_targets_from_revision(
            targets=["file.txt"],
            rev="invalid-rev",
            output=None,
            cache_dir=cache_dir,
            checkout_modes=[cache.CheckoutMode.COPY],
            force=False,
        )


def test_restore_targets_from_revision_output_with_multiple_targets(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raises GetError when -o used with multiple targets."""
    repo_path, commit = git_repo
    (repo_path / "file1.txt").write_text("content1")
    (repo_path / "file2.txt").write_text("content2")
    sha = commit("initial")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    with pytest.raises(exceptions.GetError, match="single target"):
        restore.restore_targets_from_revision(
            targets=["file1.txt", "file2.txt"],
            rev=sha[:7],
            output=repo_path / "output.txt",
            cache_dir=cache_dir,
            checkout_modes=[cache.CheckoutMode.COPY],
            force=False,
        )


def test_restore_targets_from_revision_git_file(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restores git-tracked file from revision."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original content")
    sha = commit("initial")

    # Modify file locally
    (repo_path / "file.txt").write_text("modified content")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"
    output_path = repo_path / "restored.txt"

    messages = restore.restore_targets_from_revision(
        targets=["file.txt"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert len(messages) == 1
    assert "Restored" in messages[0]
    assert output_path.read_text() == "original content"


def test_restore_targets_from_revision_skip_existing(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skips existing files without --force."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original")
    sha = commit("initial")

    # Create output file
    output_path = repo_path / "output.txt"
    output_path.write_text("existing content")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    messages = restore.restore_targets_from_revision(
        targets=["file.txt"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert len(messages) == 1
    assert "Skipped" in messages[0]
    assert output_path.read_text() == "existing content"


def test_restore_targets_from_revision_force_overwrite(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overwrites existing files with --force."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original")
    sha = commit("initial")

    # Create output file
    output_path = repo_path / "output.txt"
    output_path.write_text("existing content")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    messages = restore.restore_targets_from_revision(
        targets=["file.txt"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=True,
    )

    assert len(messages) == 1
    assert "Restored" in messages[0]
    assert output_path.read_text() == "original"


def test_restore_targets_from_revision_output_with_stage(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raises GetError when -o used with stage name."""
    repo_path, commit = git_repo

    # Create lock file for stage
    cache_dir = repo_path / ".pivot" / "cache"
    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "code_manifest": {"func:main": "abc123"},
        "params": {},
        "deps": [],
        "outs": [{"path": "model.pkl", "hash": "def456"}],
        "dep_generations": {},
    }
    (repo_path / ".pivot" / "stages" / "train.lock").write_text(yaml.dump(lock_content))

    sha = commit("add lock")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    with pytest.raises(exceptions.GetError, match="stage names"):
        restore.restore_targets_from_revision(
            targets=["train"],
            rev=sha[:7],
            output=repo_path / "output.txt",
            cache_dir=cache_dir,
            checkout_modes=[cache.CheckoutMode.COPY],
            force=False,
        )


# =============================================================================
# _out_entry_to_output_hash Tests
# =============================================================================


def test_out_entry_to_output_hash_with_manifest() -> None:
    """Converts OutEntry with manifest to OutputHash."""
    entry = {"path": "data/", "hash": "abc123", "manifest": {"a.txt": "def456"}}
    result = restore._out_entry_to_output_hash(entry)  # pyright: ignore[reportArgumentType]

    assert result is not None
    assert result["hash"] == "abc123"
    assert "manifest" in result


def test_out_entry_to_output_hash_no_hash() -> None:
    """Returns None for entry with no hash."""
    entry = {"path": "data.csv", "hash": None}
    result = restore._out_entry_to_output_hash(entry)  # pyright: ignore[reportArgumentType]

    assert result is None


# =============================================================================
# restore_file Tests
# =============================================================================


def test_restore_file_cache_miss_error(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns error message when file not in cache, git, or remote."""
    repo_path, commit = git_repo

    # Create a .pvt file that references a hash not in cache
    pvt_content = {"path": "data.csv", "hash": "nonexistent_hash_abc123", "size": 1024}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    messages = restore.restore_targets_from_revision(
        targets=["data.csv"],
        rev=sha[:7],
        output=None,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    # Function returns multiple messages: errors, blank line, summary
    assert any("not in local cache" in msg for msg in messages), (
        "Should have error about cache miss"
    )
    assert any("Failed to restore" in msg for msg in messages), "Should have failure summary"


def test_restore_file_from_cache(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Restores file from local cache when available."""
    repo_path, commit = git_repo

    # Set up cache directory
    cache_dir = repo_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)

    # Create a fake cached file
    file_hash = "abc123def456"
    cache.get_cache_path(cache_dir / "files", file_hash).parent.mkdir(parents=True, exist_ok=True)
    cache.get_cache_path(cache_dir / "files", file_hash).write_bytes(b"cached content")

    # Create a .pvt file pointing to that hash
    pvt_content = {"path": "data.csv", "hash": file_hash, "size": 14}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    output_path = repo_path / "restored_data.csv"

    messages = restore.restore_targets_from_revision(
        targets=["data.csv"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert len(messages) == 1
    assert "from cache" in messages[0]
    assert output_path.read_bytes() == b"cached content"


def test_parse_pvt_data_with_manifest() -> None:
    """Parses pvt file with manifest field."""
    content = b"""
path: data/
hash: abc123
size: 1024
manifest:
  a.txt: def456
  b.txt: ghi789
"""
    result = restore._parse_pvt_data_from_bytes(content)

    assert result is not None
    assert result["path"] == "data/"
    assert result["hash"] == "abc123"
    assert "manifest" in result
