import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from pivot import project
from pivot.storage import lock
from pivot.types import LockData

if TYPE_CHECKING:
    from pivot.types import HashInfo


def test_lock_file_creation(tmp_path: Path) -> None:
    """Lock file is created on first write."""
    stage_lock = lock.StageLock("preprocess", tmp_path)

    stage_lock.write(
        LockData(
            code_manifest={"self:preprocess": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={},
            dep_generations={},
        )
    )

    assert stage_lock.path.exists()
    assert stage_lock.path.name == "preprocess.lock"


def test_lock_file_read(set_project_root: Path) -> None:
    """Lock file contents can be read back."""
    cache_dir = set_project_root / ".cache"
    stage_lock = lock.StageLock("train", cache_dir)
    # Use absolute path for dep_hashes (internal format uses absolute paths)
    abs_data_path = str(set_project_root / "data.csv")
    dep_hashes: dict[str, HashInfo] = {abs_data_path: {"hash": "xyz123"}}
    data = LockData(
        code_manifest={"self:train": "def456", "func:helper": "ghi789"},
        params={"learning_rate": 0.01},
        dep_hashes=dep_hashes,
        output_hashes={},
        dep_generations={},
    )

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

    data = LockData(
        code_manifest=manifest,
        params={},
        dep_hashes={},
        output_hashes={},
        dep_generations={},
    )
    stage_lock.write(data)
    result = stage_lock.read()

    assert result is not None
    assert result["code_manifest"] == manifest


def test_parallel_lock_writes(tmp_path: Path) -> None:
    """Multiple stages can write locks in parallel without corruption."""
    stages = [f"stage_{i}" for i in range(10)]
    errors = list[Exception]()

    def write_lock(name: str) -> None:
        try:
            stage_lock = lock.StageLock(name, tmp_path)
            stage_lock.write(
                LockData(
                    code_manifest={f"self:{name}": f"hash_{name}"},
                    params={},
                    dep_hashes={},
                    output_hashes={},
                    dep_generations={},
                )
            )
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
        assert result == LockData(
            code_manifest={f"self:{name}": f"hash_{name}"},
            params={},
            dep_hashes={},
            output_hashes={},
            dep_generations={},
        )


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
    stage_lock = lock.StageLock("stable", tmp_path)
    fingerprint = {"self:stable": "abc", "func:helper": "def"}
    params = {"lr": 0.01}
    # In real usage, dep_hashes keys are normalized by hash_dependencies()
    normalized_key = str(project.normalize_path("data.csv"))
    dep_hashes: dict[str, HashInfo] = {normalized_key: {"hash": "xyz"}}

    stage_lock.write(
        LockData(
            code_manifest=fingerprint,
            params=params,
            dep_hashes=dep_hashes,
            output_hashes={},
            dep_generations={},
        )
    )

    changed, reason = stage_lock.is_changed(fingerprint, params, dep_hashes)

    assert changed is False
    assert reason == ""


def test_stage_changed_code_modified(tmp_path: Path) -> None:
    """Stage is marked changed when code fingerprint differs."""
    stage_lock = lock.StageLock("modified", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:modified": "old_hash"},
            params={},
            dep_hashes={},
            output_hashes={},
            dep_generations={},
        )
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
        LockData(
            code_manifest={"self:extended": "hash1"},
            params={},
            dep_hashes={},
            output_hashes={},
            dep_generations={},
        )
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
        LockData(
            code_manifest={"self:tuned": "hash"},
            params={"learning_rate": 0.01},
            dep_hashes={},
            output_hashes={},
            dep_generations={},
        )
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
    # Use absolute paths for dep_hashes to match production behavior
    input_path = str(tmp_path / "input.csv")
    old_dep_hashes: dict[str, HashInfo] = {input_path: {"hash": "old_hash"}}
    stage_lock.write(
        LockData(
            code_manifest={"self:consumer": "hash"},
            params={},
            dep_hashes=old_dep_hashes,
            output_hashes={},
            dep_generations={},
        )
    )

    new_dep_hashes: dict[str, HashInfo] = {input_path: {"hash": "new_hash"}}
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
            "output_hashes": {},
            "dep_generations": {},
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
            "output_hashes": {},
            "dep_generations": {},
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

    stage_lock.write(
        {
            "code_manifest": {"self:atomic_test": "hash"},
            "params": {},
            "dep_hashes": {},
            "output_hashes": {},
            "dep_generations": {},
        }
    )

    # Verify no .tmp file remains
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Temporary files remain: {tmp_files}"


def test_lock_directory_created(tmp_path: Path) -> None:
    """Lock file parent directories are created automatically."""
    stages_dir = tmp_path / "deep" / "nested" / "stages"
    stage_lock = lock.StageLock("nested_stage", stages_dir)

    stage_lock.write(
        {
            "code_manifest": {},
            "params": {},
            "dep_hashes": {},
            "output_hashes": {},
            "dep_generations": {},
        }
    )

    assert stage_lock.path.exists()
    assert stage_lock.path.parent == stages_dir


@pytest.mark.parametrize(
    "invalid_name",
    [
        pytest.param("", id="empty"),
        pytest.param("../etc/passwd", id="path_traversal"),
        pytest.param("stage/nested", id="nested_path"),
        pytest.param("stage with spaces", id="spaces"),
        pytest.param("stage\nwith\nnewlines", id="newlines"),
        pytest.param("../../traversal", id="double_traversal"),
    ],
)
def test_invalid_stage_name_rejected(tmp_path: Path, invalid_name: str) -> None:
    """Stage names with path traversal or special chars are rejected."""
    with pytest.raises(ValueError, match="Invalid stage name"):
        lock.StageLock(invalid_name, tmp_path)


@pytest.mark.parametrize(
    "valid_name",
    [
        pytest.param("preprocess", id="simple"),
        pytest.param("train_model", id="underscore"),
        pytest.param("evaluate-v2", id="dash"),
        pytest.param("Stage123", id="mixed_case_digits"),
        pytest.param("a", id="single_char"),
        pytest.param("A-B_C", id="mixed_separators"),
        pytest.param("stage.with.dots", id="dots"),
        pytest.param("plot@0.5", id="at_decimal"),
        pytest.param("wrangle@swe_bench", id="at_underscore"),
    ],
)
def test_valid_stage_names_accepted(tmp_path: Path, valid_name: str) -> None:
    """Valid stage names with alphanumeric, underscore, dash, dot, @ are accepted."""
    stage_lock = lock.StageLock(valid_name, tmp_path)
    assert stage_lock.stage_name == valid_name


def test_write_failure_no_orphaned_tmp(tmp_path: Path) -> None:
    """Write failure cleans up temporary file."""
    stage_lock = lock.StageLock("failing", tmp_path)

    with (
        mock.patch("yaml.dump", side_effect=RuntimeError("dump failed")),
        pytest.raises(RuntimeError, match="dump failed"),
    ):
        stage_lock.write(
            {
                "code_manifest": {},
                "params": {},
                "dep_hashes": {},
                "output_hashes": {},
                "dep_generations": {},
            }
        )

    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Orphaned temp files: {tmp_files}"


def test_concurrent_same_stage_writes(tmp_path: Path) -> None:
    """Concurrent writes to same stage don't corrupt each other."""
    errors = list[Exception]()
    results = list[int]()

    def write_value(value: int) -> None:
        try:
            stage_lock = lock.StageLock("shared", tmp_path)
            stage_lock.write(
                {
                    "code_manifest": {},
                    "params": {"thread_id": value},
                    "dep_hashes": {},
                    "output_hashes": {},
                    "dep_generations": {},
                }
            )
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
    assert final["params"]["thread_id"] in range(20), (
        "Final value should be from one of the threads"
    )

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


@pytest.mark.parametrize(
    ("missing_key", "lock_content"),
    [
        ("code_manifest", "params: {}\ndeps: []\nouts: []\ndep_generations: {}\n"),
        ("params", "code_manifest: {}\ndeps: []\nouts: []\ndep_generations: {}\n"),
        ("deps", "code_manifest: {}\nparams: {}\nouts: []\ndep_generations: {}\n"),
        ("outs", "code_manifest: {}\nparams: {}\ndeps: []\ndep_generations: {}\n"),
        ("dep_generations", "code_manifest: {}\nparams: {}\ndeps: []\nouts: []\n"),
    ],
)
def test_is_changed_with_missing_required_key_triggers_rerun(
    tmp_path: Path, missing_key: str, lock_content: str
) -> None:
    """Lock file missing any required key triggers re-run."""
    stage_lock = lock.StageLock("missing", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(lock_content)

    changed, reason = stage_lock.is_changed(
        current_fingerprint={},
        current_params={},
        dep_hashes={},
    )

    # Missing required key - read() returns None, treated as no previous run
    assert changed is True, f"Missing {missing_key} should trigger re-run"
    assert "no previous run" in reason.lower()


def test_is_changed_with_null_values_triggers_rerun(tmp_path: Path) -> None:
    """Lock file with null values triggers re-run (corrupted data)."""
    stage_lock = lock.StageLock("nulls", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    # All required keys present but with null values
    stage_lock.path.write_text(
        "code_manifest: null\nparams: null\ndeps: null\nouts: null\ndep_generations: null\n"
    )

    changed, reason = stage_lock.is_changed(
        current_fingerprint={},
        current_params={},
        dep_hashes={},
    )

    # Null values are not valid - read() returns None, treated as no previous run
    assert changed is True
    assert "no previous run" in reason.lower()


def test_read_lock_with_extra_keys_accepted(tmp_path: Path) -> None:
    """Lock files with extra keys (forward compatibility) are accepted."""
    stage_lock = lock.StageLock("future", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(
        "code_manifest: {}\nparams: {}\ndeps: []\nouts: []\n"
        + "dep_generations: {}\nfuture_key: some_value\nanother_new_field: 123\n"
    )

    result = stage_lock.read()

    assert result is not None
    assert result["code_manifest"] == {}
    assert result["params"] == {}


# =============================================================================
# Output Path Change Detection Tests
# =============================================================================


def test_stage_changed_output_path_added(tmp_path: Path) -> None:
    """Stage is marked changed when a new output path is added."""
    stage_lock = lock.StageLock("producer", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:producer": "hash"},
            params={},
            dep_hashes={},
            output_hashes={"/path/to/output.csv": {"hash": "abc123"}},
            dep_generations={},
        )
    )

    changed, reason = stage_lock.is_changed_with_lock_data(
        lock_data=stage_lock.read(),
        current_fingerprint={"self:producer": "hash"},
        current_params={},
        dep_hashes={},
        out_paths=["/path/to/output.csv", "/path/to/new_output.csv"],
    )

    assert changed is True
    assert "output" in reason.lower()


def test_stage_changed_output_path_removed(tmp_path: Path) -> None:
    """Stage is marked changed when an output path is removed."""
    stage_lock = lock.StageLock("producer", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:producer": "hash"},
            params={},
            dep_hashes={},
            output_hashes={
                "/path/to/output1.csv": {"hash": "abc123"},
                "/path/to/output2.csv": {"hash": "def456"},
            },
            dep_generations={},
        )
    )

    changed, reason = stage_lock.is_changed_with_lock_data(
        lock_data=stage_lock.read(),
        current_fingerprint={"self:producer": "hash"},
        current_params={},
        dep_hashes={},
        out_paths=["/path/to/output1.csv"],
    )

    assert changed is True
    assert "output" in reason.lower()


def test_stage_changed_output_path_modified(tmp_path: Path) -> None:
    """Stage is marked changed when output path changes (e.g., out_path_overrides)."""
    stage_lock = lock.StageLock("producer", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:producer": "hash"},
            params={},
            dep_hashes={},
            output_hashes={"/path/to/output.csv": {"hash": "abc123"}},
            dep_generations={},
        )
    )

    changed, reason = stage_lock.is_changed_with_lock_data(
        lock_data=stage_lock.read(),
        current_fingerprint={"self:producer": "hash"},
        current_params={},
        dep_hashes={},
        out_paths=["/path/to/results/output.csv"],  # Different path
    )

    assert changed is True
    assert "output" in reason.lower()


def test_stage_unchanged_with_same_output_paths(tmp_path: Path) -> None:
    """Stage is not changed when output paths match (order-independent)."""
    stage_lock = lock.StageLock("producer", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:producer": "hash"},
            params={},
            dep_hashes={},
            output_hashes={
                "/path/to/output1.csv": {"hash": "abc123"},
                "/path/to/output2.csv": {"hash": "def456"},
            },
            dep_generations={},
        )
    )

    # Provide paths in different order to test order-independence
    changed, reason = stage_lock.is_changed_with_lock_data(
        lock_data=stage_lock.read(),
        current_fingerprint={"self:producer": "hash"},
        current_params={},
        dep_hashes={},
        out_paths=["/path/to/output2.csv", "/path/to/output1.csv"],
    )

    assert changed is False
    assert reason == ""


def test_stage_unchanged_when_out_paths_none(tmp_path: Path) -> None:
    """Backward compat: when out_paths not passed, skip output path check."""
    stage_lock = lock.StageLock("producer", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:producer": "hash"},
            params={},
            dep_hashes={},
            output_hashes={"/path/to/output.csv": {"hash": "abc123"}},
            dep_generations={},
        )
    )

    # Call without out_paths parameter (backward compat)
    changed, reason = stage_lock.is_changed_with_lock_data(
        lock_data=stage_lock.read(),
        current_fingerprint={"self:producer": "hash"},
        current_params={},
        dep_hashes={},
    )

    assert changed is False
    assert reason == ""


def test_dep_path_change_invalidates_cache(tmp_path: Path) -> None:
    """Verify dep path changes correctly invalidate cache (documentation test)."""
    stage_lock = lock.StageLock("consumer", tmp_path)
    stage_lock.write(
        LockData(
            code_manifest={"self:consumer": "hash"},
            params={},
            dep_hashes={"/old/path/data.csv": {"hash": "abc123"}},
            output_hashes={},
            dep_generations={},
        )
    )

    # Same hash, different path - should invalidate
    new_dep_hashes: dict[str, HashInfo] = {"/new/path/data.csv": {"hash": "abc123"}}
    changed, reason = stage_lock.is_changed_with_lock_data(
        lock_data=stage_lock.read(),
        current_fingerprint={"self:consumer": "hash"},
        current_params={},
        dep_hashes=new_dep_hashes,
    )

    assert changed is True
    assert "input" in reason.lower() or "dep" in reason.lower()
