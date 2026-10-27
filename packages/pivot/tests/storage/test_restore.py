from __future__ import annotations

import multiprocessing
import pathlib
import queue
from typing import TYPE_CHECKING, cast

import pytest
import yaml

from pivot import exceptions, project
from pivot.cli import checkout
from pivot.storage import cache, ownership, restore, track
from pivot.types import FileHash

if TYPE_CHECKING:
    from collections.abc import Mapping
    from multiprocessing.synchronize import Event

    from conftest import GitRepo


def _hold_checkout_restore(
    project_root: pathlib.Path,
    path: pathlib.Path,
    output_hash: FileHash,
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    started: Event,
    release: Event,
    result_queue: multiprocessing.Queue[tuple[str, str]],
) -> None:
    """Pause checkout after it has acquired its restore lock."""
    project._project_root_cache = project_root
    original_checkout = cache._checkout_with_fallback

    def _checkout_after_release(
        target: pathlib.Path,
        source: pathlib.Path,
        checkout_modes: list[cache.CheckoutMode],
        *,
        executable: bool = False,
        staging_path: pathlib.Path | None = None,
        keep_existing_copy: bool = False,
    ) -> None:
        _ = staging_path
        started.set()
        assert release.wait(10)
        original_checkout(
            target,
            source,
            checkout_modes,
            executable=executable,
            keep_existing_copy=keep_existing_copy,
        )

    cache._checkout_with_fallback = _checkout_after_release
    try:
        outcome = checkout._restore_path_sync(
            path,
            output_hash,
            cache_dir,
            [cache.CheckoutMode.COPY],
            checkout.CheckoutBehavior.FORCE,
            state_dir,
        )
        result_queue.put(("checkout", outcome.result))
    finally:
        cache._checkout_with_fallback = original_checkout


def _get_while_checkout_restores(
    project_root: pathlib.Path,
    path: pathlib.Path,
    output_hash: FileHash,
    cache_dir: pathlib.Path,
    result_queue: multiprocessing.Queue[tuple[str, str]],
) -> None:
    """Run the local-cache branch of ``pivot get`` in a separate process."""
    project._project_root_cache = project_root
    result = restore.restore_file(
        rel_path=str(path.relative_to(project_root)),
        output_hash=output_hash,
        rev="HEAD",
        dest_path=path,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=True,
    )
    result_queue.put(("get", result["status"]))


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
  - producer: source
    key: data.csv
    hash: de4560000000000f
outs:
  - key: null
    hash: 0789000000000000
"""
    result = restore._parse_lock_data_from_bytes(content)

    assert result is not None
    assert "outs" in result
    out_entry = cast("dict[str, object]", cast("object", result["outs"][0]))
    assert out_entry == {"key": None, "hash": "0789000000000000"}


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
    """A stage target resolves to its outputs' workspace paths and the hashes its lock records."""
    repo_path, commit = git_repo

    # Create lock file for stage
    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "code_manifest": {"func:main": "abc123"},
        "params": {},
        "deps": [],
        "outs": [{"key": None, "hash": "de45600000000000", "tag": "data"}],
    }
    (repo_path / ".pivot" / "stages" / "train.lock").write_text(yaml.dump(lock_content))

    sha = commit("add lock file")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    stage_targets = {
        "train": restore.StageTarget(
            stage_name="train",
            state_dir=repo_path / ".pivot",
            paths_by_key={None: "data/train.pkl"},
        )
    }

    targets = restore.resolve_targets(["train"], sha, stage_targets)

    assert len(targets) == 1
    assert targets[0]["target_type"] == "stage"
    assert targets[0]["original_target"] == "train"
    assert targets[0]["paths"] == ["data/train.pkl"]
    assert targets[0]["hashes"] == {"data/train.pkl": {"hash": "de45600000000000"}}


def test_resolve_targets_stage_without_lock_at_revision(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered stage that had not run at the revision is reported, not treated as a file."""
    repo_path, commit = git_repo
    (repo_path / "dummy.txt").write_text("dummy")
    sha = commit("initial commit")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    stage_targets = {
        "train": restore.StageTarget(
            stage_name="train",
            state_dir=repo_path / ".pivot",
            paths_by_key={None: "data/train.pkl"},
        )
    }

    with pytest.raises(exceptions.TargetNotFoundError, match="Stage 'train' has no lock file"):
        restore.resolve_targets(["train"], sha, stage_targets)


def test_resolve_targets_as_pvt_file(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves target as pvt tracked file."""
    repo_path, commit = git_repo

    # Create pvt file
    pvt_content = {"path": "data.csv", "hash": "abc123", "size": 1024}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    targets = restore.resolve_targets(["data.csv"], sha, {})

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

    targets = restore.resolve_targets(["readme.txt"], sha, {})

    assert len(targets) == 1
    assert targets[0]["target_type"] == "file"
    assert targets[0]["hashes"]["readme.txt"] is None  # No hash for git-only files


def test_resolve_targets_not_found(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises TargetNotFoundError for unknown tarrestore."""
    repo_path, commit = git_repo

    (repo_path / "dummy.txt").write_text("dummy")
    sha = commit("initial commit")[:7]

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    with pytest.raises(exceptions.TargetNotFoundError, match="nonexistent"):
        restore.resolve_targets(["nonexistent"], sha, {})


def test_resolve_targets_path_traversal(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises TargetNotFoundError for path traversal in tarrestore."""
    repo_path, commit = git_repo

    (repo_path / "dummy.txt").write_text("dummy")
    sha = commit("initial commit")[:7]

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    with pytest.raises(exceptions.TargetNotFoundError, match="Path traversal"):
        restore.resolve_targets(["../../../etc/passwd"], sha, {})


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
            stage_targets={},
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
            stage_targets={},
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

    messages, _ = restore.restore_targets_from_revision(
        targets=["file.txt"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={},
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

    messages, _ = restore.restore_targets_from_revision(
        targets=["file.txt"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={},
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

    messages, _ = restore.restore_targets_from_revision(
        targets=["file.txt"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=True,
    )

    assert len(messages) == 1
    assert "Restored" in messages[0]
    assert output_path.read_text() == "original"


def test_restore_targets_from_revision_output_with_stage(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single stage output can be restored to an explicit --output path."""
    repo_path, commit = git_repo
    cache_dir = repo_path / ".pivot" / "cache"
    source = repo_path / "cached-stage-output.txt"
    source.write_text("stage output")
    output_hash = cache.save_to_cache(
        source, cache_dir / "files", checkout_mode=cache.CheckoutMode.COPY
    )
    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "code_manifest": {"func:main": "abc123"},
        "params": {},
        "deps": [],
        "outs": [
            {
                "key": None,
                "hash": output_hash["hash"],
                "path": "data/train/output.txt",
            }
        ],
    }
    (repo_path / ".pivot" / "stages" / "train.lock").write_text(yaml.dump(lock_content))
    sha = commit("add lock")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "output.txt"

    messages, success = restore.restore_targets_from_revision(
        targets=["train"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={
            "train": restore.StageTarget(
                stage_name="train", state_dir=repo_path / ".pivot", paths_by_key={None: "model.pkl"}
            )
        },
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert success
    assert messages == [f"Restored: {output_path} (from cache)"]
    assert output_path.read_text() == "stage output"


@pytest.mark.parametrize(
    "output_arg",
    [
        pytest.param(None, id="workspace-output"),
        pytest.param(pathlib.Path("restored-tree"), id="output-flag"),
    ],
)
def test_restore_targets_from_revision_force_refuses_output_holding_tracked_file(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, output_arg: pathlib.Path | None
) -> None:
    """`pivot get --force` refuses before it removes a tracked file in a stage output or in an
    explicit `--output` destination (relative to the working directory)."""
    repo_path, commit = git_repo
    cache_dir = repo_path / ".pivot" / "cache"
    source = repo_path / "cached-stage-output"
    source.mkdir()
    (source / "generated.txt").write_text("generated")
    output_hash = cache.save_to_cache(
        source, cache_dir / "files", checkout_mode=cache.CheckoutMode.COPY
    )
    assert "manifest" in output_hash
    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "code_manifest": {"func:main": "abc123"},
        "params": {},
        "deps": [],
        "outs": [
            {
                "key": None,
                "hash": output_hash["hash"],
                "manifest": output_hash["manifest"],
                "path": "data/b/tree",
            }
        ],
    }
    (repo_path / ".pivot" / "stages" / "tree.lock").write_text(yaml.dump(lock_content))
    sha = commit("add lock")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    destination = (
        repo_path / output_arg if output_arg is not None else repo_path / "data" / "b" / "tree"
    )
    tracked = destination / "saved.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("user input")
    pvt_path = track.get_pvt_path(tracked)
    track.write_pvt_file(
        pvt_path, track.PvtData(path="saved.txt", hash="abc1230000000000", size=10)
    )

    with pytest.raises(exceptions.PivotError, match=r"saved\.txt\.pvt"):
        restore.restore_targets_from_revision(
            targets=["tree"],
            rev=sha[:7],
            output=output_arg,
            cache_dir=cache_dir,
            stage_targets={
                "tree": restore.StageTarget(
                    stage_name="tree", state_dir=repo_path / ".pivot", paths_by_key={}
                )
            },
            checkout_modes=[cache.CheckoutMode.COPY],
            force=True,
        )

    assert tracked.read_text() == "user input"
    assert pvt_path.exists()


def test_restore_file_target_force_relative_output_refuses_tracked_directory(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative `pivot get --force -o` (from the working directory) cannot replace a directory
    holding tracked input."""
    repo_path, commit = git_repo
    source = repo_path / "source.txt"
    source.write_text("source data")
    source_hash = cache.save_to_cache(
        source, repo_path / ".pivot" / "cache" / "files", checkout_mode=cache.CheckoutMode.COPY
    )
    track.write_pvt_file(
        track.get_pvt_path(source),
        track.PvtData(path="source.txt", hash=source_hash["hash"], size=len("source data")),
    )
    sha = commit("add tracked source")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    destination = repo_path / "data" / "b" / "tree"
    saved = destination / "saved.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("user input")
    saved_pvt = track.get_pvt_path(saved)
    track.write_pvt_file(
        saved_pvt, track.PvtData(path="saved.txt", hash="abc1230000000000", size=10)
    )

    with pytest.raises(exceptions.PivotError, match=r"saved\.txt\.pvt"):
        restore.restore_targets_from_revision(
            targets=["source.txt"],
            rev=sha[:7],
            output=pathlib.Path("data/b/tree"),
            cache_dir=repo_path / ".pivot" / "cache",
            stage_targets={},
            checkout_modes=[cache.CheckoutMode.COPY],
            force=True,
        )

    assert saved.read_text() == "user input"
    assert saved_pvt.exists()


# =============================================================================
# restore_file Tests
# =============================================================================


def test_restore_file_cache_miss_error(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns error message when file not in cache, git, or remote."""
    repo_path, commit = git_repo

    # Create a .pvt file that references a hash not in cache
    pvt_content = {"path": "data.csv", "hash": "deadbeef12345678", "size": 1024}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    messages, _ = restore.restore_targets_from_revision(
        targets=["data.csv"],
        rev=sha[:7],
        output=None,
        cache_dir=cache_dir,
        stage_targets={},
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
    file_hash = "abc123def4567890"
    cache.get_cache_path(cache_dir / "files", file_hash).parent.mkdir(parents=True, exist_ok=True)
    cache.get_cache_path(cache_dir / "files", file_hash).write_bytes(b"cached content")

    # Create a .pvt file pointing to that hash
    pvt_content = {"path": "data.csv", "hash": file_hash, "size": 14}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    output_path = repo_path / "restored_data.csv"

    messages, _ = restore.restore_targets_from_revision(
        targets=["data.csv"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert len(messages) == 1
    assert "from cache" in messages[0]
    assert output_path.read_bytes() == b"cached content"


def test_get_waits_for_checkout_restore_on_shared_cache(tmp_path: pathlib.Path) -> None:
    """Get and checkout share a restore lock when the cache is outside the project."""
    project_root = tmp_path / "project"
    state_dir = project_root / ".pivot"
    state_dir.mkdir(parents=True)
    shared_cache = tmp_path / "shared" / "cache"
    file_hash = "0123456789abcdef"
    cache_path = cache.get_cache_path(shared_cache / "files", file_hash)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached content")
    target = project_root / "data" / "out.txt"
    target.parent.mkdir(parents=True)
    output_hash = FileHash(hash=file_hash)

    ctx = multiprocessing.get_context("spawn")
    started = ctx.Event()
    release = ctx.Event()
    results = ctx.Queue()
    checkout_proc = ctx.Process(
        target=_hold_checkout_restore,
        args=(
            project_root,
            target,
            output_hash,
            shared_cache / "files",
            state_dir,
            started,
            release,
            results,
        ),
    )
    checkout_proc.start()
    assert started.wait(10)

    get_proc = ctx.Process(
        target=_get_while_checkout_restores,
        args=(project_root, target, output_hash, shared_cache, results),
    )
    get_proc.start()
    with pytest.raises(queue.Empty):
        results.get(timeout=0.5)

    release.set()
    checkout_proc.join(10)
    get_proc.join(10)

    assert checkout_proc.exitcode == 0
    assert get_proc.exitcode == 0
    assert {results.get(timeout=1), results.get(timeout=1)} == {
        ("checkout", "restored"),
        ("get", restore.RestoreStatus.RESTORED),
    }
    assert target.read_text() == "cached content"
    assert cache._get_lock_path(target, state_dir).exists()


def test_restore_file_force_keeps_existing_output_until_cache_swap(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forced cache restore replaces an existing output atomically under its lock."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    cache_dir = tmp_path / "cache"
    file_hash = "0123456789abcdef"
    cache_path = cache.get_cache_path(cache_dir / "files", file_hash)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached content")
    target = tmp_path / "data" / "out.txt"
    target.parent.mkdir()
    target.write_text("existing content")

    def _unexpected_remove(path: pathlib.Path) -> None:
        raise AssertionError(f"cache restore removed {path} before its atomic swap")

    monkeypatch.setattr(cache, "remove_output", _unexpected_remove)

    result = restore.restore_file(
        rel_path="data/out.txt",
        output_hash=FileHash(hash=file_hash),
        rev="HEAD",
        dest_path=target,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=True,
    )

    assert result["status"] is restore.RestoreStatus.RESTORED
    assert target.read_text() == "cached content"


def test_restore_file_records_pvt_members_before_writing_them(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pivot get` of a stage restores the revision's output, which no lock here records: it
    records each `*.pvt` member the restore writes, with the content it will hold, before it
    writes anything; a target it skips records nothing."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    cache_dir = tmp_path / "cache"
    source = tmp_path / "source"
    source.mkdir()
    (source / "notes.pvt").write_text("stage notes")
    (source / "value.txt").write_text("v")
    output_hash = cache.save_to_cache(source, cache_dir / "files")
    target = tmp_path / "data" / "b" / "tree"
    recorded = list[dict[pathlib.Path, str]]()

    def record(members: Mapping[pathlib.Path, str]) -> None:
        assert not target.exists()
        recorded.append(dict(members))

    result = restore.restore_file(
        rel_path="data/b/tree",
        output_hash=output_hash,
        rev="HEAD",
        dest_path=target,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
        record_members=record,
    )

    assert result["status"] is restore.RestoreStatus.RESTORED, result["message"]
    notes = target / "notes.pvt"
    assert recorded == [{ownership.normalized(notes): cache.hash_file(notes)[0]}]
    assert notes.read_text() == "stage notes"

    skipped = restore.restore_file(
        rel_path="data/b/tree",
        output_hash=output_hash,
        rev="HEAD",
        dest_path=target,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
        record_members=record,
    )

    assert skipped["status"] is restore.RestoreStatus.SKIPPED
    assert len(recorded) == 1


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


# =============================================================================
# Cache Permission Tests
# =============================================================================


def test_restore_directory_from_cache_with_manifest(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restores directory from cache when manifest files are present (not tree hash)."""
    repo_path, commit = git_repo

    # Set up cache directory
    cache_dir = repo_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)

    # Create cached files (individual file hashes, NOT the tree hash)
    file1_hash = "1111111111111111"
    file2_hash = "2222222222222222"
    cache.get_cache_path(cache_dir / "files", file1_hash).parent.mkdir(parents=True, exist_ok=True)
    cache.get_cache_path(cache_dir / "files", file1_hash).write_bytes(b"file 1 content")
    cache.get_cache_path(cache_dir / "files", file2_hash).parent.mkdir(parents=True, exist_ok=True)
    cache.get_cache_path(cache_dir / "files", file2_hash).write_bytes(b"file 2 content")

    # Create a .pvt file for a directory with manifest
    # Tree hash is different from file hashes - it's a hash of the manifest JSON
    tree_hash = "aaaaaaaaaaaaaaaa"
    manifest = [
        {"relpath": "a.txt", "hash": file1_hash, "size": 14, "isexec": False},
        {"relpath": "b.txt", "hash": file2_hash, "size": 14, "isexec": False},
    ]
    pvt_content = {"path": "data/", "hash": tree_hash, "size": 28, "manifest": manifest}
    (repo_path / "data.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    output_path = repo_path / "restored_data"

    messages, success = restore.restore_targets_from_revision(
        targets=["data"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert success, f"Expected success, got messages: {messages}"
    assert len(messages) == 1
    assert "from cache" in messages[0]
    assert (output_path / "a.txt").read_bytes() == b"file 1 content"
    assert (output_path / "b.txt").read_bytes() == b"file 2 content"


def test_restore_file_from_remote_caches_with_readonly_permissions(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files cached from remote have read-only (0o444) permissions."""
    import xxhash

    repo_path, commit = git_repo

    # Set up directories
    cache_dir = repo_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)

    # Create content that will "come from remote"
    content = b"content from remote"
    file_hash = xxhash.xxh64(content).hexdigest()

    # Create a .pvt file pointing to that hash (not in local cache)
    pvt_content = {"path": "data.csv", "hash": file_hash, "size": len(content)}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Mock remote.fetch_from_remote to return our content
    monkeypatch.setattr(
        "pivot.storage.restore.remote.fetch_from_remote",
        lambda h: content,  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    )

    output_path = repo_path / "restored_data.csv"

    messages, success = restore.restore_targets_from_revision(
        targets=["data.csv"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert success
    assert len(messages) == 1
    assert "from remote" in messages[0]
    assert output_path.read_bytes() == content

    # Verify cached file has read-only permissions
    cached_path = cache.get_cache_path(cache_dir / "files", file_hash)
    assert cached_path.exists(), "File should be cached after restore from remote"
    mode = cached_path.stat().st_mode & 0o777
    assert mode == 0o444, f"Cache file should have 0o444 permissions, got {oct(mode)}"


def test_restore_file_from_remote_keeps_an_object_already_cached(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caching remote content never replaces an object a concurrent commit published first:
    a checkout may be linking or symlinking it, and its executable bit belongs to it."""
    import xxhash

    repo_path, commit = git_repo
    cache_dir = repo_path / ".pivot" / "cache"
    content = b"#!/bin/sh\necho from remote\n"
    file_hash = xxhash.xxh64(content).hexdigest()
    (repo_path / "tool.pvt").write_text(
        yaml.dump({"path": "tool", "hash": file_hash, "size": len(content)})
    )
    sha = commit("add pvt file")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    cached_path = cache.get_cache_path(cache_dir / "files", file_hash)

    def _fetch_while_a_commit_publishes(_hash: str) -> bytes:
        cached_path.parent.mkdir(parents=True)
        cached_path.write_bytes(content)
        cached_path.chmod(0o555)
        return content

    monkeypatch.setattr(
        "pivot.storage.restore.remote.fetch_from_remote", _fetch_while_a_commit_publishes
    )

    _, success = restore.restore_targets_from_revision(
        targets=["tool"],
        rev=sha[:7],
        output=repo_path / "restored_tool",
        cache_dir=cache_dir,
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert success
    assert cached_path.stat().st_mode & 0o777 == 0o555, "the published object was replaced"
    left = [p.name for p in cached_path.parent.iterdir() if p.name != cache._PREFIX_LOCK]
    assert left == [cached_path.name], "no temp file left"


def test_restore_file_from_remote_hash_mismatch(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote content with hash mismatch is rejected and returns error."""
    import xxhash

    repo_path, commit = git_repo

    # Set up directories
    cache_dir = repo_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)

    # Create content but calculate hash for DIFFERENT content
    actual_content = b"corrupted content from remote"
    expected_content = b"original content"
    expected_hash = xxhash.xxh64(expected_content).hexdigest()

    # Create a .pvt file pointing to the expected hash (not in local cache)
    pvt_content = {"path": "data.csv", "hash": expected_hash, "size": len(expected_content)}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Mock remote.fetch_from_remote to return WRONG content (simulating corruption)
    monkeypatch.setattr(
        "pivot.storage.restore.remote.fetch_from_remote",
        lambda h: actual_content,  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    )

    output_path = repo_path / "restored_data.csv"

    messages, success = restore.restore_targets_from_revision(
        targets=["data.csv"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    # Should fail due to hash mismatch
    assert not success
    assert any("corrupted" in msg or "hash mismatch" in msg for msg in messages), (
        f"Should have hash mismatch error, got: {messages}"
    )
    # Output file should NOT be created with corrupted content
    assert not output_path.exists(), "Corrupted content should not be written to output"
    # Cache should NOT store corrupted content
    cached_path = cache.get_cache_path(cache_dir / "files", expected_hash)
    assert not cached_path.exists(), "Corrupted content should not be cached"


def test_restore_file_from_remote_success(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote content is restored and cached successfully."""
    import xxhash

    repo_path, commit = git_repo

    # Set up directories
    cache_dir = repo_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)

    content = b"content from remote"
    file_hash = xxhash.xxh64(content).hexdigest()

    pvt_content = {"path": "data.csv", "hash": file_hash, "size": len(content)}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.setattr(
        "pivot.storage.restore.remote.fetch_from_remote",
        lambda h: content,  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    )

    # Create output directory as read-only to simulate write failure
    output_dir = repo_path / "readonly_dir"
    output_dir.mkdir()
    output_path = output_dir / "data.csv"

    # First restore should succeed
    messages, success = restore.restore_targets_from_revision(
        targets=["data.csv"],
        rev=sha[:7],
        output=output_path,
        cache_dir=cache_dir,
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
    )

    assert success
    assert output_path.read_bytes() == content
