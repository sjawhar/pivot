"""Tests for per-stage lock files."""

import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from pivot import lock

if TYPE_CHECKING:
    from pivot.types import HashInfo, LockData


def test_lock_file_creation(tmp_path: Path) -> None:
    """Lock file is created on first write."""
    stage_lock = lock.StageLock("preprocess", tmp_path)

    stage_lock.write({"code_manifest": {"self:preprocess": "abc123"}})

    assert stage_lock.path.exists()
    assert stage_lock.path.name == "preprocess.lock"


def test_lock_file_read(tmp_path: Path) -> None:
    """Lock file contents can be read back."""
    stage_lock = lock.StageLock("train", tmp_path)
    dep_hashes: dict[str, HashInfo] = {"data.csv": {"hash": "xyz123"}}
    data: LockData = {
        "code_manifest": {"self:train": "def456", "func:helper": "ghi789"},
        "params": {"learning_rate": 0.01},
        "dep_hashes": dep_hashes,
    }

    stage_lock.write(data)
    result = stage_lock.read()

    assert result == data


def test_lock_file_read_missing(tmp_path: Path) -> None:
    """Reading non-existent lock file returns None."""
    stage_lock = lock.StageLock("missing", tmp_path)

    result = stage_lock.read()

    assert result is None


def test_manifest_preservation(tmp_path: Path) -> None:
    """Code manifest is preserved exactly through write/read cycle."""
    stage_lock = lock.StageLock("evaluate", tmp_path)
    manifest = {
        "self:evaluate": "hash1",
        "func:compute_metrics": "hash2",
        "func:load_model": "hash3",
        "mod:sklearn.metrics": "hash4",
        "const:THRESHOLD": "hash5",
    }

    data: LockData = {"code_manifest": manifest}
    stage_lock.write(data)
    result = stage_lock.read()

    assert result is not None
    assert result.get("code_manifest") == manifest


def test_parallel_lock_writes(tmp_path: Path) -> None:
    """Multiple stages can write locks in parallel without corruption."""
    stages = [f"stage_{i}" for i in range(10)]
    errors = list[Exception]()

    def write_lock(name: str) -> None:
        try:
            stage_lock = lock.StageLock(name, tmp_path)
            stage_lock.write({"code_manifest": {f"self:{name}": f"hash_{name}"}})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_lock, args=(name,)) for name in stages]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during parallel writes: {errors}"

    for name in stages:
        stage_lock = lock.StageLock(name, tmp_path)
        result = stage_lock.read()
        assert result == {"code_manifest": {f"self:{name}": f"hash_{name}"}}


def test_stage_changed_no_previous_run(tmp_path: Path) -> None:
    """Stage is marked changed when no lock file exists."""
    stage_lock = lock.StageLock("new_stage", tmp_path)

    changed, reason = stage_lock.is_changed(
        current_fingerprint={"self:new_stage": "abc"},
        current_params={},
        dep_hashes={},
    )

    assert changed is True
    assert "no previous run" in reason.lower()


def test_stage_unchanged_when_identical(tmp_path: Path) -> None:
    """Stage is not changed when fingerprint, params, and deps match."""
    from pivot import project

    stage_lock = lock.StageLock("stable", tmp_path)
    fingerprint = {"self:stable": "abc", "func:helper": "def"}
    params = {"lr": 0.01}
    # In real usage, dep_hashes keys are normalized by hash_dependencies()
    normalized_key = str(project.normalize_path("data.csv"))
    dep_hashes: dict[str, HashInfo] = {normalized_key: {"hash": "xyz"}}

    stage_lock.write(
        {
            "code_manifest": fingerprint,
            "params": params,
            "dep_hashes": dep_hashes,
        }
    )

    changed, reason = stage_lock.is_changed(fingerprint, params, dep_hashes)

    assert changed is False
    assert reason == ""


def test_stage_changed_code_modified(tmp_path: Path) -> None:
    """Stage is marked changed when code fingerprint differs."""
    stage_lock = lock.StageLock("modified", tmp_path)
    stage_lock.write(
        {
            "code_manifest": {"self:modified": "old_hash"},
            "params": {},
            "dep_hashes": {},
        }
    )

    changed, reason = stage_lock.is_changed(
        current_fingerprint={"self:modified": "new_hash"},
        current_params={},
        dep_hashes={},
    )

    assert changed is True
    assert "code changed" in reason.lower()


def test_stage_changed_new_dependency(tmp_path: Path) -> None:
    """Stage is marked changed when new code dependency added."""
    stage_lock = lock.StageLock("extended", tmp_path)
    stage_lock.write(
        {
            "code_manifest": {"self:extended": "hash1"},
            "params": {},
            "dep_hashes": {},
        }
    )

    changed, reason = stage_lock.is_changed(
        current_fingerprint={"self:extended": "hash1", "func:new_helper": "hash2"},
        current_params={},
        dep_hashes={},
    )

    assert changed is True
    assert "code changed" in reason.lower()


def test_stage_changed_params_modified(tmp_path: Path) -> None:
    """Stage is marked changed when params differ."""
    stage_lock = lock.StageLock("tuned", tmp_path)
    stage_lock.write(
        {
            "code_manifest": {"self:tuned": "hash"},
            "params": {"learning_rate": 0.01},
            "dep_hashes": {},
        }
    )

    changed, reason = stage_lock.is_changed(
        current_fingerprint={"self:tuned": "hash"},
        current_params={"learning_rate": 0.001},
        dep_hashes={},
    )

    assert changed is True
    assert "params changed" in reason.lower()


def test_stage_changed_dep_hash_modified(tmp_path: Path) -> None:
    """Stage is marked changed when input file hash differs."""
    stage_lock = lock.StageLock("consumer", tmp_path)
    old_dep_hashes: dict[str, HashInfo] = {"input.csv": {"hash": "old_hash"}}
    stage_lock.write(
        {
            "code_manifest": {"self:consumer": "hash"},
            "params": {},
            "dep_hashes": old_dep_hashes,
        }
    )

    new_dep_hashes: dict[str, HashInfo] = {"input.csv": {"hash": "new_hash"}}
    changed, reason = stage_lock.is_changed(
        current_fingerprint={"self:consumer": "hash"},
        current_params={},
        dep_hashes=new_dep_hashes,
    )

    assert changed is True
    assert "input" in reason.lower() or "dep" in reason.lower()


def test_stage_changed_dep_added(tmp_path: Path) -> None:
    """Stage is marked changed when new input dependency added."""
    stage_lock = lock.StageLock("consumer", tmp_path)
    old_dep_hashes: dict[str, HashInfo] = {"a.csv": {"hash": "hash_a"}}
    stage_lock.write(
        {
            "code_manifest": {"self:consumer": "hash"},
            "params": {},
            "dep_hashes": old_dep_hashes,
        }
    )

    new_dep_hashes: dict[str, HashInfo] = {
        "a.csv": {"hash": "hash_a"},
        "b.csv": {"hash": "hash_b"},
    }
    changed, reason = stage_lock.is_changed(
        current_fingerprint={"self:consumer": "hash"},
        current_params={},
        dep_hashes=new_dep_hashes,
    )

    assert changed is True


def test_stage_changed_dep_removed(tmp_path: Path) -> None:
    """Stage is marked changed when input dependency removed."""
    stage_lock = lock.StageLock("consumer", tmp_path)
    old_dep_hashes: dict[str, HashInfo] = {
        "a.csv": {"hash": "hash_a"},
        "b.csv": {"hash": "hash_b"},
    }
    stage_lock.write(
        {
            "code_manifest": {"self:consumer": "hash"},
            "params": {},
            "dep_hashes": old_dep_hashes,
        }
    )

    new_dep_hashes: dict[str, HashInfo] = {"a.csv": {"hash": "hash_a"}}
    changed, reason = stage_lock.is_changed(
        current_fingerprint={"self:consumer": "hash"},
        current_params={},
        dep_hashes=new_dep_hashes,
    )

    assert changed is True


def test_atomic_write_no_partial_file(tmp_path: Path) -> None:
    """Write failure should not leave partial lock file."""
    stage_lock = lock.StageLock("atomic_test", tmp_path)

    stage_lock.write({"code_manifest": {"self:atomic_test": "hash"}})

    # Verify no .tmp file remains
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Temporary files remain: {tmp_files}"


def test_lock_directory_created(tmp_path: Path) -> None:
    """Lock file parent directories are created automatically."""
    nested_cache = tmp_path / "deep" / "nested" / "cache"
    stage_lock = lock.StageLock("nested_stage", nested_cache)

    stage_lock.write({"code_manifest": {}})

    assert stage_lock.path.exists()
    assert stage_lock.path.parent == nested_cache / "stages"


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "../etc/passwd",
        "stage/nested",
        "stage.with.dots",
        "stage with spaces",
        "stage\nwith\nnewlines",
        "../../traversal",
    ],
)
def test_invalid_stage_name_rejected(tmp_path: Path, invalid_name: str) -> None:
    """Stage names with path traversal or special chars are rejected."""
    with pytest.raises(ValueError, match="Invalid stage name"):
        lock.StageLock(invalid_name, tmp_path)


@pytest.mark.parametrize(
    "valid_name",
    [
        "preprocess",
        "train_model",
        "evaluate-v2",
        "Stage123",
        "a",
        "A-B_C",
    ],
)
def test_valid_stage_names_accepted(tmp_path: Path, valid_name: str) -> None:
    """Valid stage names with alphanumeric, underscore, dash are accepted."""
    stage_lock = lock.StageLock(valid_name, tmp_path)
    assert stage_lock.stage_name == valid_name


def test_write_failure_no_orphaned_tmp(tmp_path: Path) -> None:
    """Write failure cleans up temporary file."""
    stage_lock = lock.StageLock("failing", tmp_path)

    with (
        mock.patch("yaml.dump", side_effect=RuntimeError("dump failed")),
        pytest.raises(RuntimeError, match="dump failed"),
    ):
        stage_lock.write({"code_manifest": {}})

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Orphaned temp files: {tmp_files}"


def test_concurrent_same_stage_writes(tmp_path: Path) -> None:
    """Concurrent writes to same stage don't corrupt each other."""
    errors = list[Exception]()
    results = list[int]()

    def write_value(value: int) -> None:
        try:
            stage_lock = lock.StageLock("shared", tmp_path)
            # Use valid LockData with params key to track which thread won
            stage_lock.write({"params": {"thread_id": value}})
            results.append(value)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_value, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent writes: {errors}"

    stage_lock = lock.StageLock("shared", tmp_path)
    final = stage_lock.read()
    assert final is not None
    params = final.get("params")
    assert params is not None, "params should be present in lock data"
    assert params["thread_id"] in range(20), "Final value should be from one of the threads"

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Orphaned temp files: {tmp_files}"


def test_read_corrupted_non_dict_returns_none(tmp_path: Path) -> None:
    """Lock file with non-dict YAML returns None (treated as missing)."""
    stage_lock = lock.StageLock("corrupted", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("just a string\n")

    result = stage_lock.read()

    assert result is None


def test_read_corrupted_list_returns_none(tmp_path: Path) -> None:
    """Lock file with list YAML returns None."""
    stage_lock = lock.StageLock("corrupted", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("- item1\n- item2\n")

    result = stage_lock.read()

    assert result is None


def test_read_binary_garbage_returns_none(tmp_path: Path) -> None:
    """Lock file with binary garbage returns None."""
    stage_lock = lock.StageLock("binary", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_bytes(b"\xff\xfe\x00\x01\x80\x81")

    result = stage_lock.read()

    assert result is None


def test_read_invalid_yaml_returns_none(tmp_path: Path) -> None:
    """Lock file with invalid YAML syntax returns None."""
    stage_lock = lock.StageLock("invalid", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("key: [unclosed bracket\n")

    result = stage_lock.read()

    assert result is None


def test_read_empty_file_returns_none(tmp_path: Path) -> None:
    """Empty lock file returns None."""
    stage_lock = lock.StageLock("empty", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("")

    result = stage_lock.read()

    assert result is None


def test_is_changed_handles_explicit_null_values(tmp_path: Path) -> None:
    """Lock file with explicit null values treated as empty dict."""
    stage_lock = lock.StageLock("nulls", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text("code_manifest: null\nparams: null\ndep_hashes: null\n")

    changed, reason = stage_lock.is_changed(
        current_fingerprint={},
        current_params={},
        dep_hashes={},
    )

    assert changed is False, f"Should not be changed but got: {reason}"
