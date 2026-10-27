from __future__ import annotations

import errno
import multiprocessing
import os
import queue
import re
from typing import TYPE_CHECKING, cast

import pytest
import xxhash
import yaml

from pivot import exceptions, project, types
from pivot.cli import checkout
from pivot.remote import sync as transfer
from pivot.storage import cache, listing, lock, restore, track
from pivot.types import DirHash, DirManifestEntry, FileHash, TransferResult

if TYPE_CHECKING:
    import pathlib
    from multiprocessing.synchronize import Event

    from conftest import GitRepo
    from pivot.remote.protocol import RemoteStorage


def _helper_unused_remote() -> RemoteStorage:
    raise AssertionError("this restore must not open the remote")


# A directory with a nested file and an empty directory, as its listing records it.
_DIR_CONTENTS = {"a.txt": b"file 1 content", "sub/b.txt": b"file 2 content"}
_DIR_EMPTY = ["empty"]


def _cache_directory_files(
    cache_files: pathlib.Path, contents: dict[str, bytes], empty_dirs: list[str]
) -> listing.HashedDirectory:
    """Put each file of a directory in the cache and return the directory as hashed.

    The directory's listing is not stored: a test commits it or caches it as it needs.
    """
    files = list[DirManifestEntry]()
    for relpath, content in contents.items():
        file_hash = xxhash.xxh64(content).hexdigest()
        cache_path = cache.get_cache_path(cache_files, file_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
        files.append(
            DirManifestEntry(relpath=relpath, hash=file_hash, size=len(content), isexec=False)
        )
    members = listing.DirListing(files=files, empty_dirs=empty_dirs)
    data = listing.encode(members)
    return listing.HashedDirectory(
        record=listing.record_of(members, data), listing=members, data=data
    )


def _write_directory_pvt(pvt_path: pathlib.Path, record: DirHash) -> pathlib.Path:
    """Write a tracked directory's ``.pvt`` (its record only, no listing file)."""
    track.write_pvt_file(
        pvt_path,
        track.PvtData(
            path=pvt_path.name.removesuffix(".pvt"),
            hash=record["hash"],
            size=record["size"],
            nfiles=record["nfiles"],
        ),
    )
    return pvt_path


def _restore_data_directory(
    repo_path: pathlib.Path, rev: str, output_path: pathlib.Path, *, force: bool = False
) -> tuple[list[str], bool]:
    """Restore tracked directory ``data`` at ``rev`` to ``output_path`` from the repo's cache."""
    return restore.restore_targets_from_revision(
        targets=["data"],
        rev=rev,
        output=output_path,
        cache_dir=repo_path / ".pivot" / "cache",
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=force,
        remote_factory=_helper_unused_remote,
    )


def _tree(root: pathlib.Path) -> list[str]:
    """Every file and directory under ``root``, as sorted POSIX relpaths."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


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
    ) -> os.stat_result:
        _ = staging_path
        started.set()
        assert release.wait(10)
        return original_checkout(
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
            checkout.Expected(output_hash),
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
        remote_factory=_helper_unused_remote,
    )
    result_queue.put(("get", result["status"]))


# =============================================================================
# _parse_lock_data_from_bytes Tests
# =============================================================================


def test_parse_lock_data_from_bytes_valid() -> None:
    """Parses valid lock file content."""
    content = b"""
schema_version: 3
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
    assert result.listings == {}
    out_entry = cast("dict[str, object]", cast("object", result.data["outs"][0]))
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
schema_version: 3
code_manifest:
  func:main: abc123
"""
    result = restore._parse_lock_data_from_bytes(content)

    assert result is None


def test_parse_lock_data_from_bytes_reads_an_older_schema_whole() -> None:
    """A past revision keeps the lock format an older pivot wrote: its directory entry, which
    listed the files inline, becomes this format's record, and the inline files its listing."""
    content = b"""
schema_version: 2
code_manifest:
  func:main: abc123
params: {}
deps: []
outs:
  - key: null
    hash: 0789000000000000
    path: data/p/out
    manifest:
      - relpath: a.txt
        hash: "1111111111111111"
        size: 14
        isexec: false
    empty_dirs: [logs]
"""
    result = restore._parse_lock_data_from_bytes(content)

    assert result is not None
    members = listing.DirListing(
        files=[
            types.DirManifestEntry(relpath="a.txt", hash="1111111111111111", size=14, isexec=False)
        ],
        empty_dirs=["logs"],
    )
    assert result.listings == {None: members}
    assert result.data["schema_version"] == 3
    assert result.data["outs"] == [
        {
            "key": None,
            "hash": listing.digest(listing.encode(members)),
            "path": "data/p/out",
            "size": 14,
            "nfiles": 1,
        }
    ]


@pytest.mark.parametrize(
    "relpath",
    [pytest.param("../escape.txt", id="traversal"), pytest.param("/etc/passwd", id="absolute")],
)
def test_parse_lock_data_from_bytes_refuses_an_older_inline_listing_outside_its_directory(
    relpath: str,
) -> None:
    content = f"""
schema_version: 2
code_manifest: {{}}
params: {{}}
deps: []
outs:
  - key: null
    hash: 0789000000000000
    manifest:
      - relpath: {relpath}
        hash: "1111111111111111"
        size: 14
        isexec: false
""".encode()

    assert restore._parse_lock_data_from_bytes(content) is None


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

    assert result == ({"path": "data.csv", "hash": "abc123", "size": 1024}, None), (
        "A file .pvt has no listing"
    )


def test_parse_pvt_data_from_bytes_directory_record() -> None:
    """A directory .pvt holds only its record; its members are in its listing file."""
    content = b"""
path: data
hash: 0123456789abcdef
size: 28
nfiles: 2
"""
    result = restore._parse_pvt_data_from_bytes(content)

    assert result == (
        {"path": "data", "hash": "0123456789abcdef", "size": 28, "nfiles": 2},
        None,
    ), "A directory .pvt's listing is read from <path>.pvt.listing, not the .pvt"


def test_parse_pvt_data_from_bytes_older_inline_listing() -> None:
    """An older pivot's directory .pvt is read as the record of the listing it holds inline."""
    members = listing.DirListing(
        files=[
            DirManifestEntry(relpath="a.txt", hash="1111111111111111", size=14, isexec=False),
            DirManifestEntry(relpath="sub/b.txt", hash="2222222222222222", size=14, isexec=True),
        ],
        empty_dirs=["empty"],
    )
    old_document = {
        "path": "data",
        "hash": "aaaaaaaaaaaaaaaa",
        "size": 28,
        "num_files": 2,
        "manifest": members["files"],
        "empty_dirs": members["empty_dirs"],
    }

    result = restore._parse_pvt_data_from_bytes(yaml.dump(old_document).encode())

    assert result is not None
    data, inline = result
    assert inline == members
    assert data == {
        "path": "data",
        "hash": listing.digest(listing.encode(members)),
        "size": 28,
        "nfiles": 2,
    }, "The record is the inline listing's, not the hash the older pivot wrote"


@pytest.mark.parametrize(
    ("relpath", "empty_dir"),
    [
        pytest.param("../outside.txt", "empty", id="file-in-parent"),
        pytest.param("/etc/passwd", "empty", id="absolute-file"),
        pytest.param("a.txt", "../outside", id="empty-dir-in-parent"),
    ],
)
def test_parse_pvt_data_from_bytes_older_inline_listing_outside_directory(
    relpath: str, empty_dir: str
) -> None:
    """Returns None for an older pivot's directory .pvt naming a path outside the directory."""
    old_document = {
        "path": "data",
        "hash": "aaaaaaaaaaaaaaaa",
        "size": 14,
        "num_files": 1,
        "manifest": [{"relpath": relpath, "hash": "1111111111111111", "size": 14, "isexec": False}],
        "empty_dirs": [empty_dir],
    }

    result = restore._parse_pvt_data_from_bytes(yaml.dump(old_document).encode())

    assert result is None


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


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"path: data.csv\nhash: abc123\nsize: '1'\n", id="size-not-an-int"),
        pytest.param(b"path: data.csv\nhash: abc123\nsize: -1\n", id="negative-size"),
        pytest.param(b"path: data\nhash: abc123\nsize: 1\nnfiles: -1\n", id="negative-nfiles"),
        pytest.param(b"hash: abc123\nsize: 1\nmanifest: []\n", id="older-without-path"),
    ],
)
def test_parse_pvt_data_from_bytes_wrong_field_types(content: bytes) -> None:
    """Returns None, not a crash, for a value is_pvt_data rejects: a non-int or negative count,
    or an older inline listing with no path."""
    assert restore._parse_pvt_data_from_bytes(content) is None


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        pytest.param(
            b"path: data.csv\nhash: 1234567890123456\nsize: 1\n",
            "field 'hash' is not a string",
            id="all-digit-hash",
        ),
        pytest.param(
            b"path: 1234\nhash: abc123\nsize: 1\n", "field 'path' is not a string", id="int-path"
        ),
        pytest.param(
            b"path: data/\nhash: abc123\nsize: 1\nmanifest:\n- relpath: a.txt\n  isexec: false\n",
            "field 'manifest[0].hash' is missing",
            id="manifest-entry-without-hash",
        ),
    ],
)
def test_parse_pvt_data_from_bytes_names_a_field_a_restore_cannot_use(
    content: bytes, reason: str
) -> None:
    """A .pvt whose field a restore builds a path or cache key from has the wrong type, or whose
    manifest entry lacks a field a restore reads, returns the reason naming the field."""
    assert restore._parse_pvt_data_from_bytes(content) == reason


# =============================================================================
# resolve_targets Tests
# =============================================================================


def test_resolve_targets_as_stage(git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage target resolves to its outputs' workspace paths and the hashes its lock records."""
    repo_path, commit = git_repo

    # Create lock file for stage
    (repo_path / ".pivot" / "stages").mkdir(parents=True)
    lock_content = {
        "schema_version": lock.SCHEMA_VERSION,
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


def test_resolve_targets_reports_an_invalid_pvt_as_that_targets_error(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .pvt at the revision that is not valid resolves to that target's error, naming the
    file, the revision and the field, rather than stopping every target."""
    repo_path, commit = git_repo
    (repo_path / "data.csv.pvt").write_text("path: data.csv\nhash: 1234567890123456\nsize: 1\n")
    sha = commit("add pvt file")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    targets = restore.resolve_targets(["data.csv"], sha, {})

    assert targets[0]["paths"] == ["data.csv"]
    assert "error" in targets[0]
    assert targets[0]["error"] == (
        f"data.csv.pvt at {sha} is not a valid .pvt file: field 'hash' is not a string"
    )


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
            remote_factory=_helper_unused_remote,
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
            remote_factory=_helper_unused_remote,
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
        remote_factory=_helper_unused_remote,
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
        remote_factory=_helper_unused_remote,
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
        remote_factory=_helper_unused_remote,
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
        "schema_version": lock.SCHEMA_VERSION,
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
        remote_factory=_helper_unused_remote,
    )

    assert success
    assert messages == [f"Restored: {output_path} (from cache)"]
    assert output_path.read_text() == "stage output"


def test_restore_stage_directory_output_from_listing_beside_lock(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage's directory output is restored from the listing committed beside its lock: the
    lock records only the directory's hash, size and file count, and the cache holds no listing.
    """
    repo_path, commit = git_repo
    cache_dir = repo_path / ".pivot" / "cache"
    hashed = _cache_directory_files(cache_dir / "files", _DIR_CONTENTS, _DIR_EMPTY)
    record = hashed["record"]
    state_dir = repo_path / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    stages_dir.mkdir(parents=True)
    lock_content = {
        "schema_version": lock.SCHEMA_VERSION,
        "code_manifest": {"func:main": "abc123"},
        "params": {},
        "deps": [],
        "outs": [],
        "records": {
            "deps": [],
            "outs": [
                {
                    "key": None,
                    "listing": record["hash"],
                    "size": record["size"],
                    "nfiles": record["nfiles"],
                    "path": "data/train",
                }
            ],
        },
    }
    (stages_dir / "train.lock").write_text(yaml.dump(lock_content))
    listing.write(
        listing.output_listing_path(stages_dir, types.ArtifactIdentity("train", None)),
        hashed["data"],
    )
    sha = commit("add lock and listing")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "data" / "train"

    messages, success = restore.restore_targets_from_revision(
        targets=["train"],
        rev=sha[:7],
        output=None,
        cache_dir=cache_dir,
        stage_targets={
            "train": restore.StageTarget(
                stage_name="train", state_dir=state_dir, paths_by_key={None: "data/train"}
            )
        },
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
        remote_factory=_helper_unused_remote,
    )

    assert success, f"Expected success, got messages: {messages}"
    assert messages == [f"Restored: {output_path} (from cache)"]
    assert _tree(output_path) == ["a.txt", "empty", "sub", "sub/b.txt"]
    assert (output_path / "sub" / "b.txt").read_bytes() == b"file 2 content"
    assert cache.hash_directory(output_path)["record"] == record, (
        "The restored directory should hash to the record its lock holds"
    )


# =============================================================================
# restore_file Tests
# =============================================================================


def test_restore_file_cache_miss_reports_the_remote_error(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracked file in neither the cache nor git reaches the remote step, whose error names
    what is wrong with the remote: here, that none is configured."""
    repo_path, commit = git_repo

    # Create a .pvt file that references a hash not in cache
    pvt_content = {"path": "data.csv", "hash": "deadbeef12345678", "size": 1024}
    (repo_path / "data.csv.pvt").write_text(yaml.dump(pvt_content))

    sha = commit("add pvt file")

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    cache_dir = repo_path / ".pivot" / "cache"

    messages, success = restore.restore_targets_from_revision(
        targets=["data.csv"],
        rev=sha[:7],
        output=None,
        cache_dir=cache_dir,
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
        remote_factory=lambda: transfer.create_remote_from_name(None)[0],
    )

    assert not success
    assert any(
        "data.csv - failed to download from remote: No remotes configured." in msg
        for msg in messages
    ), messages
    assert any("Failed to restore" in msg for msg in messages), "Should have failure summary"


class _UnreachableRemote:
    """A remote no download can connect to; counts the download batches asked of it."""

    url: str = "s3://unreachable/cache"
    batches: int

    def __init__(self) -> None:
        self.batches = 0

    async def download_batch(
        self, items: list[tuple[str, pathlib.Path]], *, readonly: bool = False
    ) -> list[TransferResult]:
        assert readonly, "get downloads cache objects read-only"
        self.batches += 1
        return [
            TransferResult(hash=h, success=False, error="cannot connect", connection_failed=True)
            for h, _ in items
        ]


def test_restore_targets_tries_an_unreachable_remote_once(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a target's download cannot connect to the remote, each later target that needs the
    remote fails naming it, without another download."""
    repo_path, commit = git_repo
    for name, file_hash in (("a.csv", "aaaaaaaaaaaaaaaa"), ("b.csv", "bbbbbbbbbbbbbbbb")):
        pvt_content = {"path": name, "hash": file_hash, "size": 1}
        (repo_path / f"{name}.pvt").write_text(yaml.dump(pvt_content))
    sha = commit("add pvt files")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    remote = _UnreachableRemote()

    messages, success = restore.restore_targets_from_revision(
        targets=["a.csv", "b.csv"],
        rev=sha[:7],
        output=None,
        cache_dir=repo_path / ".pivot" / "cache",
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
        # The fake has only the url and download_batch a restore uses
        remote_factory=lambda: cast("RemoteStorage", cast("object", remote)),
    )

    assert not success
    assert remote.batches == 1, "the second target does not try the unreachable remote"
    assert "a.csv - failed to download from remote: cannot connect" in messages[0], messages
    assert (
        "b.csv - failed to download from remote: s3://unreachable/cache could not be reached "
        + "for an earlier target, so it is not tried again"
    ) in messages[1], messages


class _RefusingRemote:
    """A remote that answers but refuses this user's credentials for every download; counts the
    download batches asked of it."""

    url: str = "gs://refusing/cache"
    batches: int

    def __init__(self) -> None:
        self.batches = 0

    async def download_batch(
        self, items: list[tuple[str, pathlib.Path]], *, readonly: bool = False
    ) -> list[TransferResult]:
        self.batches += 1
        raise exceptions.RemoteConnectionError("credentials refused")


def test_restore_targets_tries_a_remote_that_answered_again_for_each_target(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote that answered with an error (a RemoteConnectionError that is not
    RemoteUnreachableError: credentials refused, one object denied) is tried again by each later
    target, which reports the remote's own reason."""
    repo_path, commit = git_repo
    for name, file_hash in (("a.csv", "aaaaaaaaaaaaaaaa"), ("b.csv", "bbbbbbbbbbbbbbbb")):
        pvt_content = {"path": name, "hash": file_hash, "size": 1}
        (repo_path / f"{name}.pvt").write_text(yaml.dump(pvt_content))
    sha = commit("add pvt files")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    remote = _RefusingRemote()

    messages, success = restore.restore_targets_from_revision(
        targets=["a.csv", "b.csv"],
        rev=sha[:7],
        output=None,
        cache_dir=repo_path / ".pivot" / "cache",
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.COPY],
        force=False,
        # The fake has only the url and download_batch a restore uses
        remote_factory=lambda: cast("RemoteStorage", cast("object", remote)),
    )

    assert not success
    assert remote.batches == 2, "each target tries a remote that answered"
    for message in messages[:2]:
        assert "failed to download from remote: credentials refused" in message, messages


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        pytest.param(
            'manifest:\n- relpath: "a\\0b"\n  hash: fedcba9876543210\n  size: 1\n'
            + "  isexec: false\n",
            "field 'manifest[0].relpath' holds a NUL byte",
            id="nul-relpath",
        ),
        pytest.param(
            "manifest:\n- relpath: a.txt\n  hash: fedcba9876543210\n  size: 1\n"
            + '  isexec: false\nempty_dirs:\n- "e\\0"\n',
            "field 'empty_dirs[0]' holds a NUL byte",
            id="nul-empty-dir",
        ),
    ],
)
def test_symlink_restore_reports_a_pvt_path_it_cannot_build(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, fields: str, field: str
) -> None:
    """A symlink restore through a tree no pivot has checked builds a path from each manifest
    relpath and empty directory, so one holding a NUL byte is that target's error, not a raw
    ValueError."""
    repo_path, commit = git_repo
    tree = cache.get_cache_path(repo_path / ".pivot" / "cache" / "files", "0123456789abcdef")
    tree.mkdir(parents=True)
    (tree / "a.txt").write_text("a")
    (repo_path / "d.pvt").write_text("path: d\nhash: '0123456789abcdef'\nsize: 1\n" + fields)
    sha = commit("add pvt")[:7]
    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    messages, success = restore.restore_targets_from_revision(
        targets=["d"],
        rev=sha,
        output=None,
        cache_dir=repo_path / ".pivot" / "cache",
        stage_targets={},
        checkout_modes=[cache.CheckoutMode.SYMLINK],
        force=False,
        remote_factory=_helper_unused_remote,
    )

    assert not success
    assert f"d - d.pvt at {sha} is not a valid .pvt file: {field}" in messages[0], messages


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
        remote_factory=_helper_unused_remote,
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
        remote_factory=_helper_unused_remote,
    )

    assert result["status"] is restore.RestoreStatus.RESTORED
    assert target.read_text() == "cached content"


# =============================================================================
# Directory restore Tests
# =============================================================================


def test_restore_directory_from_committed_listing(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracked directory is restored from the listing committed beside its .pvt; the cache
    holds the directory's files but no listing object."""
    repo_path, commit = git_repo
    hashed = _cache_directory_files(
        repo_path / ".pivot" / "cache" / "files", _DIR_CONTENTS, _DIR_EMPTY
    )
    pvt_path = _write_directory_pvt(repo_path / "data.pvt", hashed["record"])
    listing.write(listing.tracked_listing_path(pvt_path), hashed["data"])
    sha = commit("track data")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    messages, success = _restore_data_directory(repo_path, sha[:7], output_path)

    assert success, f"Expected success, got messages: {messages}"
    assert messages == [f"Restored: {output_path} (from cache)"]
    assert _tree(output_path) == ["a.txt", "empty", "sub", "sub/b.txt"]
    assert (output_path / "a.txt").read_bytes() == b"file 1 content"
    assert cache.hash_directory(output_path)["record"] == hashed["record"], (
        "The restored directory should hash to the record its .pvt holds"
    )


def test_restore_directory_from_cache_listing_object(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory whose listing file the revision lacks is restored from the cache's listing
    object at the directory's hash."""
    repo_path, commit = git_repo
    cache_files = repo_path / ".pivot" / "cache" / "files"
    hashed = _cache_directory_files(cache_files, _DIR_CONTENTS, _DIR_EMPTY)
    _write_directory_pvt(repo_path / "data.pvt", hashed["record"])
    listing.store_object(cache_files, hashed["record"]["hash"], hashed["data"])
    sha = commit("track data without its listing file")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    messages, success = _restore_data_directory(repo_path, sha[:7], output_path)

    assert success, f"Expected success, got messages: {messages}"
    assert messages == [f"Restored: {output_path} (from cache)"]
    assert _tree(output_path) == ["a.txt", "empty", "sub", "sub/b.txt"]
    assert cache.hash_directory(output_path)["record"] == hashed["record"], (
        "The restored directory should hash to the record its .pvt holds"
    )


def test_restore_directory_from_older_inline_pvt(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older pivot's directory .pvt, its files listed inline, is restored from that list;
    the restore leaves the .pvt as it is and writes no listing file."""
    repo_path, commit = git_repo
    hashed = _cache_directory_files(
        repo_path / ".pivot" / "cache" / "files", _DIR_CONTENTS, _DIR_EMPTY
    )
    old_document = {
        "path": "data",
        "hash": "aaaaaaaaaaaaaaaa",
        "size": 28,
        "num_files": 2,
        "manifest": hashed["listing"]["files"],
        "empty_dirs": hashed["listing"]["empty_dirs"],
    }
    pvt_path = repo_path / "data.pvt"
    pvt_path.write_text(yaml.dump(old_document))
    old_bytes = pvt_path.read_bytes()
    sha = commit("track data with an older pivot")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    messages, success = _restore_data_directory(repo_path, sha[:7], output_path)

    assert success, f"Expected success, got messages: {messages}"
    assert messages == [f"Restored: {output_path} (from cache)"]
    assert cache.hash_directory(output_path)["record"] == hashed["record"], (
        "The restored directory should hold exactly what the inline list records"
    )
    assert pvt_path.read_bytes() == old_bytes, "A restore never rewrites an older .pvt"
    assert not listing.tracked_listing_path(pvt_path).exists(), (
        "A restore never converts an older .pvt to a listing file"
    )


def test_restore_directory_without_listing_fails(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory whose listing is neither committed nor cached is not restored, though every
    file it holds is cached."""
    repo_path, commit = git_repo
    hashed = _cache_directory_files(
        repo_path / ".pivot" / "cache" / "files", _DIR_CONTENTS, _DIR_EMPTY
    )
    _write_directory_pvt(repo_path / "data.pvt", hashed["record"])
    sha = commit("track data without its listing file")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    messages, success = _restore_data_directory(repo_path, sha[:7], output_path)

    assert not success
    assert any("not in local cache" in msg for msg in messages), (
        f"Should report the directory missing, got: {messages}"
    )
    assert not output_path.exists(), "Nothing should be written without a listing"


def test_restore_directory_reads_the_cache_copy_of_a_committed_listing_not_matching_its_hash(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A committed listing whose bytes do not hash to the .pvt's directory hash is passed over
    for the cache's copy at that hash, with one warning naming the file at the revision: the
    one read rule every reader follows."""
    repo_path, commit = git_repo
    cache_files = repo_path / ".pivot" / "cache" / "files"
    hashed = _cache_directory_files(cache_files, _DIR_CONTENTS, _DIR_EMPTY)
    pvt_path = _write_directory_pvt(repo_path / "data.pvt", hashed["record"])
    edited = hashed["data"].replace(b" 14 - a.txt\n", b" 15 - a.txt\n")
    assert edited != hashed["data"]
    listing.write(listing.tracked_listing_path(pvt_path), edited)
    listing.store_object(cache_files, hashed["record"]["hash"], hashed["data"])
    sha = commit("track data with an edited listing")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    _, success = _restore_data_directory(repo_path, sha[:7], output_path)

    assert success
    assert _tree(output_path) == ["a.txt", "empty", "sub", "sub/b.txt"]
    assert (output_path / "a.txt").read_bytes() == _DIR_CONTENTS["a.txt"]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        f"data.pvt.listing at revision {sha[:7]}: its bytes hash to" in m
        and "using the cache's copy" in m
        for m in warnings
    ), warnings


def test_restore_directory_refuses_committed_listing_not_matching_its_hash_without_cache_copy(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no cache object at the .pvt's directory hash to read instead, a committed listing
    that does not hash to it is refused, naming the file at the revision, and nothing is
    written."""
    repo_path, commit = git_repo
    cache_files = repo_path / ".pivot" / "cache" / "files"
    hashed = _cache_directory_files(cache_files, _DIR_CONTENTS, _DIR_EMPTY)
    pvt_path = _write_directory_pvt(repo_path / "data.pvt", hashed["record"])
    edited = hashed["data"].replace(b" 14 - a.txt\n", b" 15 - a.txt\n")
    listing.write(listing.tracked_listing_path(pvt_path), edited)
    sha = commit("track data with an edited listing")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    with pytest.raises(
        exceptions.CorruptListingError,
        match=re.escape(f"data.pvt.listing at revision {sha[:7]}: its bytes hash to"),
    ):
        _restore_data_directory(repo_path, sha[:7], output_path)

    assert not output_path.exists(), "Nothing should be written from a corrupt listing"


def test_restore_directory_refuses_corrupt_cache_listing_object(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache listing object whose bytes do not hash to the directory's hash is refused,
    naming the object, and nothing is written."""
    repo_path, commit = git_repo
    cache_files = repo_path / ".pivot" / "cache" / "files"
    hashed = _cache_directory_files(cache_files, _DIR_CONTENTS, _DIR_EMPTY)
    _write_directory_pvt(repo_path / "data.pvt", hashed["record"])
    sha = commit("track data without its listing file")
    cached_listing = cache.get_cache_path(cache_files, hashed["record"]["hash"])
    cached_listing.parent.mkdir(parents=True, exist_ok=True)
    cached_listing.write_bytes(hashed["data"].replace(b" 14 - a.txt\n", b" 15 - a.txt\n"))
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    with pytest.raises(
        exceptions.CorruptListingError, match=re.escape(f"{cached_listing}: its bytes hash to")
    ):
        _restore_data_directory(repo_path, sha[:7], output_path)

    assert not output_path.exists(), "Nothing should be written from a corrupt listing"


def test_restore_directory_refuses_listing_disagreeing_with_record(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing that hashes to the .pvt's directory hash but lists a file count other than the
    one the .pvt records is refused, and nothing is written."""
    repo_path, commit = git_repo
    hashed = _cache_directory_files(
        repo_path / ".pivot" / "cache" / "files", _DIR_CONTENTS, _DIR_EMPTY
    )
    record = hashed["record"]
    pvt_path = _write_directory_pvt(
        repo_path / "data.pvt",
        DirHash(hash=record["hash"], size=record["size"], nfiles=record["nfiles"] + 1),
    )
    listing.write(listing.tracked_listing_path(pvt_path), hashed["data"])
    sha = commit("track data with an edited .pvt")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    with pytest.raises(
        exceptions.CorruptListingError,
        match=re.escape("lists 2 files of 28 bytes, but its lock or .pvt records 3 files"),
    ):
        _restore_data_directory(repo_path, sha[:7], output_path)

    assert not output_path.exists(), "Nothing should be written from a corrupt listing"


@pytest.mark.parametrize(
    ("member", "error"),
    [
        pytest.param("../outside.txt", "an empty, '.' or '..' segment", id="parent-segment"),
        pytest.param("/outside.txt", "an absolute path", id="absolute"),
    ],
)
def test_restore_directory_refuses_listing_naming_path_outside_it(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, member: str, error: str
) -> None:
    """A listing naming a file outside its directory is refused before anything is written,
    though its bytes hash to the directory hash the .pvt records."""
    repo_path, commit = git_repo
    content = b"outside content"
    file_hash = xxhash.xxh64(content).hexdigest()
    cache_path = cache.get_cache_path(repo_path / ".pivot" / "cache" / "files", file_hash)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(content)
    data = f"{file_hash} {len(content)} - {member}\n".encode()
    pvt_path = _write_directory_pvt(
        repo_path / "data.pvt", DirHash(hash=listing.digest(data), size=len(content), nfiles=1)
    )
    listing.write(listing.tracked_listing_path(pvt_path), data)
    sha = commit("track data with a listing naming an outside path")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"

    with pytest.raises(exceptions.CorruptListingError, match=re.escape(error)):
        _restore_data_directory(repo_path, sha[:7], output_path)

    assert not (repo_path / "outside.txt").exists(), "No file should be written outside"
    assert not output_path.exists(), "Nothing should be written from an unsafe listing"


def test_restore_directory_force_replaces_existing_directory_with_listed_members(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forced restore swaps in exactly the listed members for an existing directory, leaving
    no temporary or backup copy beside it."""
    repo_path, commit = git_repo
    hashed = _cache_directory_files(
        repo_path / ".pivot" / "cache" / "files", _DIR_CONTENTS, _DIR_EMPTY
    )
    pvt_path = _write_directory_pvt(repo_path / "data.pvt", hashed["record"])
    listing.write(listing.tracked_listing_path(pvt_path), hashed["data"])
    sha = commit("track data")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    output_path = repo_path / "restored_data"
    output_path.mkdir()
    (output_path / "a.txt").write_text("local edit")
    (output_path / "stale.txt").write_text("not listed")
    siblings = sorted(path.name for path in repo_path.iterdir())

    messages, success = _restore_data_directory(repo_path, sha[:7], output_path, force=True)

    assert success, f"Expected success, got messages: {messages}"
    assert _tree(output_path) == ["a.txt", "empty", "sub", "sub/b.txt"]
    assert cache.hash_directory(output_path)["record"] == hashed["record"], (
        "The restored directory should hold exactly what its listing records"
    )
    assert sorted(path.name for path in repo_path.iterdir()) == siblings, (
        "No temporary or backup copy should be left beside the directory"
    )


def test_restore_file_labels_a_retried_cache_restore_from_cache(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache restore that fails once is retried by the remote step; with nothing to download,
    a retry that succeeds is reported as from the cache, not from the remote."""
    repo_path, commit = git_repo
    (repo_path / "README").write_text("not the restored file")
    commit("initial")
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    cache_dir = repo_path / ".pivot" / "cache"
    file_hash = "0123456789abcdef"
    cache_path = cache.get_cache_path(cache_dir / "files", file_hash)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached content")
    target = repo_path / "data" / "out.txt"
    real_link = os.link
    failed = list[str]()

    def _link_failing_once(src: pathlib.Path, dst: pathlib.Path) -> None:
        if not failed:
            failed.append(str(dst))
            raise OSError(errno.EIO, "injected I/O error", str(dst))
        real_link(src, dst)

    monkeypatch.setattr(os, "link", _link_failing_once)

    result = restore.restore_file(
        rel_path="data/out.txt",
        output_hash=FileHash(hash=file_hash),
        rev="HEAD",
        dest_path=target,
        cache_dir=cache_dir,
        checkout_modes=[cache.CheckoutMode.HARDLINK],
        force=False,
        remote_factory=_helper_unused_remote,
    )

    assert failed, "the first hard link should have failed"
    assert result["status"] is restore.RestoreStatus.RESTORED, result["message"]
    assert result["message"] == f"Restored: {target} (from cache)"
    assert target.read_text() == "cached content"


def test_parse_pvt_data_rejects_a_manifest_that_is_not_a_list() -> None:
    """A manifest written as a mapping of path to hash is not the list of entries .pvt records,
    so it is reported instead of failing a restore that reads its entries."""
    content = b"path: data/\nhash: abc123\nsize: 1024\nmanifest:\n  a.txt: def456\n"

    assert restore._parse_pvt_data_from_bytes(content) == "field 'manifest' is not a list"
