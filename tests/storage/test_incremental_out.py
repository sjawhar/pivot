import pathlib
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

import pytest

from helpers import register_test_stage
from pivot import exceptions, executor, loaders, outputs
from pivot.executor import worker
from pivot.registry import REGISTRY
from pivot.storage import cache, lock
from pivot.types import FileHash, LockData

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _IncrementalStageOutputs(TypedDict):
    database: Annotated[pathlib.Path, outputs.Out("database.txt", loaders.PathOnly())]


class _RegularStageOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


# =============================================================================
# Module-level stage functions for testing (must be picklable)
# =============================================================================


def _incremental_stage_append() -> _IncrementalStageOutputs:
    """Stage that appends to an incremental output."""
    import pathlib

    db_path = pathlib.Path("database.txt")
    if db_path.exists():
        existing = db_path.read_text()
        count = len(existing.strip().split("\n")) if existing.strip() else 0
    else:
        existing = ""
        count = 0

    with open(db_path, "w") as f:
        f.write(existing)
        f.write(f"line {count + 1}\n")

    return {"database": db_path}


def _regular_stage_create() -> _RegularStageOutputs:
    """Stage that creates a regular output."""
    pathlib.Path("output.txt").write_text("created\n")
    return {"output": pathlib.Path("output.txt")}


# =============================================================================
# Test helper for IncrementalOut registration
# =============================================================================


def _register_incremental_stage(
    func: object,
    name: str,
    out_path: str,
) -> None:
    """Register a stage and convert its Out to IncrementalOut for testing.

    This is needed because the annotation system uses outputs.Out, but
    IncrementalOut is a separate outputs.IncrementalOut type that requires
    special handling during execution.
    """
    # Register normally (annotations create outputs.Out)
    register_test_stage(cast("Callable[..., Any]", func), name=name)

    # Replace the Out with IncrementalOut in the registry
    # This is a test-only hack to test IncrementalOut behavior
    stage_info = REGISTRY._stages[name]
    stage_info["outs"] = [outputs.IncrementalOut(path=out_path, loader=loaders.PathOnly())]
    stage_info["outs_paths"] = [out_path]


# =============================================================================
# Prepare Outputs for Execution Tests
# =============================================================================


def test_prepare_outputs_regular_out_is_deleted(tmp_path: pathlib.Path) -> None:
    """Regular Out should be deleted before execution."""
    output_file = tmp_path / "output.txt"
    output_file.write_text("existing content")

    stage_outs: list[outputs.Out[Any]] = [
        outputs.Out(path=str(output_file), loader=loaders.PathOnly())
    ]
    worker._prepare_outputs_for_execution(stage_outs, None, tmp_path / "cache")

    assert not output_file.exists()


def test_prepare_outputs_incremental_no_cache_creates_empty(tmp_path: pathlib.Path) -> None:
    """IncrementalOut with no cache should start fresh (file doesn't exist)."""
    output_file = tmp_path / "database.txt"

    stage_outs: list[outputs.Out[Any]] = [
        outputs.IncrementalOut(path=str(output_file), loader=loaders.PathOnly())
    ]
    worker._prepare_outputs_for_execution(stage_outs, None, tmp_path / "cache")

    assert not output_file.exists()


def test_prepare_outputs_incremental_restores_from_cache(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IncrementalOut should restore from cache before execution."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "database.txt"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create a cached version
    output_file.write_text("cached content\n")
    output_hash = cache.save_to_cache(output_file, cache_dir)

    # Lock data simulating previous run (uses relative path like production)
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={"database.txt": output_hash},
        dep_generations={},
    )

    # Prepare for execution (uses relative path like production)
    stage_outs = [outputs.IncrementalOut(path="database.txt", loader=loaders.PathOnly())]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # File should be restored
    assert output_file.exists()
    assert output_file.read_text() == "cached content\n"


def test_prepare_outputs_incremental_restored_file_is_writable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restored IncrementalOut should be a writable copy, not symlink."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "database.txt"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create a cached version
    output_file.write_text("cached content\n")
    output_hash = cache.save_to_cache(output_file, cache_dir)

    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={"database.txt": output_hash},
        dep_generations={},
    )

    # Prepare for execution (uses relative path like production)
    stage_outs = [outputs.IncrementalOut(path="database.txt", loader=loaders.PathOnly())]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # Should NOT be a symlink (should be a copy)
    assert not output_file.is_symlink()

    # Should be writable
    output_file.write_text("modified content\n")
    assert output_file.read_text() == "modified content\n"


def test_prepare_outputs_incremental_missing_cache_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IncrementalOut with lock data but missing cache should raise clear error."""
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Lock data referencing a hash that doesn't exist in cache (16 chars)
    fake_hash = FileHash(hash="abcd1234abcd1234")
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={"database.txt": fake_hash},
        dep_generations={},
    )

    stage_outs: list[outputs.Out[Any]] = [
        outputs.IncrementalOut(path="database.txt", loader=loaders.PathOnly())
    ]

    with pytest.raises(exceptions.CacheRestoreError) as exc_info:
        worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    assert "Cache missing for IncrementalOut 'database.txt'" in str(exc_info.value)


@pytest.mark.parametrize(
    ("no_commit", "expected_lock_dir"),
    [
        pytest.param(False, ".pivot/stages", id="production_lock"),
        pytest.param(True, ".pivot/pending/stages", id="pending_lock"),
    ],
)
def test_integration_missing_cache_error_includes_recovery_suggestion(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_registry: None,
    no_commit: bool,
    expected_lock_dir: str,
) -> None:
    """Full error message should include correct lock path and recovery suggestions."""
    import shutil

    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "database.txt"
    cache_dir = tmp_path / ".pivot" / "cache"

    _register_incremental_stage(
        _incremental_stage_append,
        name="append_stage",
        out_path=str(db_path),
    )

    # First run creates and caches the output
    results1 = executor.run(cache_dir=cache_dir, no_commit=no_commit)
    assert results1["append_stage"]["status"] == "ran"

    # Delete the cache files AND output file (but keep the lock file)
    # This simulates: lock says "file had hash X" but cache is missing
    shutil.rmtree(cache_dir / "files")
    db_path.unlink()

    # Second run with force should fail with helpful error message
    results2 = executor.run(cache_dir=cache_dir, force=True, no_commit=no_commit)
    assert results2["append_stage"]["status"] == "failed"

    reason = results2["append_stage"]["reason"]
    assert "Cache missing for IncrementalOut" in reason
    assert "pivot pull" in reason
    assert expected_lock_dir in reason
    assert "append_stage.lock" in reason


# =============================================================================
# IncrementalOut DVC Export Tests
# =============================================================================


def test_dvc_export_incremental_out_always_persist() -> None:
    """IncrementalOut should always export with persist: true."""
    from pivot import dvc_compat

    inc = outputs.IncrementalOut(path="database.csv", loader=loaders.PathOnly())
    result = dvc_compat._build_out_entry(inc, "database.csv")
    assert result == {"database.csv": {"persist": True}}


def test_dvc_export_incremental_out_with_cache_false() -> None:
    """IncrementalOut with cache=False should export both options."""
    from pivot import dvc_compat

    inc = outputs.IncrementalOut(path="database.csv", loader=loaders.PathOnly(), cache=False)
    result = dvc_compat._build_out_entry(inc, "database.csv")
    assert result == {"database.csv": {"cache": False, "persist": True}}


# =============================================================================
# IncrementalOut Integration Tests
# =============================================================================


def test_integration_first_run_creates_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """First run with IncrementalOut should create the output from scratch."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "database.txt"

    _register_incremental_stage(
        _incremental_stage_append,
        name="append_stage",
        out_path=str(db_path),
    )

    results = executor.run(cache_dir=tmp_path / ".pivot" / "cache")

    assert results["append_stage"]["status"] == "ran"
    assert db_path.exists()
    assert db_path.read_text() == "line 1\n"


def test_integration_second_run_appends_to_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """Second run should restore and append to existing output."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "database.txt"
    cache_dir = tmp_path / ".pivot" / "cache"
    state_dir = tmp_path / ".pivot"

    _register_incremental_stage(
        _incremental_stage_append,
        name="append_stage",
        out_path=str(db_path),
    )

    # First run
    executor.run(cache_dir=cache_dir)
    assert db_path.read_text() == "line 1\n"

    # Simulate code change by modifying the lock file's code_manifest
    # Keep output_hashes so we can restore
    stage_lock = lock.StageLock("append_stage", lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    assert lock_data is not None
    lock_data["code_manifest"] = {"self:fake": "changed_hash"}
    stage_lock.write(lock_data)

    # Delete the output file to verify restoration works
    db_path.unlink()

    # Second run - should restore from cache and append
    results = executor.run(cache_dir=cache_dir)

    assert results["append_stage"]["status"] == "ran"
    assert db_path.read_text() == "line 1\nline 2\n"


# =============================================================================
# IncrementalOut Directory Tests
# =============================================================================


def test_incremental_out_restores_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IncrementalOut should restore directory from cache with COPY mode."""
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "data_dir"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create directory with nested structure
    output_dir.mkdir()
    (output_dir / "file1.txt").write_text("content1")
    subdir = output_dir / "subdir"
    subdir.mkdir()
    (subdir / "file2.txt").write_text("content2")

    # Save to cache
    output_hash = cache.save_to_cache(output_dir, cache_dir)

    # Simulate lock data from previous run (uses relative path like production)
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={"data_dir": output_hash},
        dep_generations={},
    )

    # Delete the output
    cache.remove_output(output_dir)
    assert not output_dir.exists()

    # Prepare for execution (restore with COPY mode, uses relative path)
    stage_outs = [outputs.IncrementalOut(path="data_dir", loader=loaders.PathOnly())]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # Directory should be restored
    assert output_dir.exists()
    assert (output_dir / "file1.txt").read_text() == "content1"
    assert (output_dir / "subdir" / "file2.txt").read_text() == "content2"


def test_incremental_out_directory_is_writable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restored directory should allow creating new files."""
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "data_dir"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create directory
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("existing")

    # Save to cache
    output_hash = cache.save_to_cache(output_dir, cache_dir)

    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={"data_dir": output_hash},
        dep_generations={},
    )

    # Delete and restore (uses relative path like production)
    cache.remove_output(output_dir)
    stage_outs = [outputs.IncrementalOut(path="data_dir", loader=loaders.PathOnly())]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # Should be able to write new files
    new_file = output_dir / "new_file.txt"
    new_file.write_text("new content")
    assert new_file.read_text() == "new content"

    # Should be able to modify existing files
    (output_dir / "existing.txt").write_text("modified")
    assert (output_dir / "existing.txt").read_text() == "modified"


def test_incremental_out_directory_subdirs_writable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restored subdirectories should allow creating new files."""
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "data_dir"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create directory with nested structure
    output_dir.mkdir()
    subdir = output_dir / "subdir"
    subdir.mkdir()
    (subdir / "existing.txt").write_text("existing")

    # Save to cache
    output_hash = cache.save_to_cache(output_dir, cache_dir)

    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={"data_dir": output_hash},
        dep_generations={},
    )

    # Delete and restore (uses relative path like production)
    cache.remove_output(output_dir)
    stage_outs = [outputs.IncrementalOut(path="data_dir", loader=loaders.PathOnly())]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # Should be able to create files in subdirectories
    new_file = subdir / "new_in_subdir.txt"
    new_file.write_text("new content in subdir")
    assert new_file.read_text() == "new content in subdir"


# =============================================================================
# Executable Bit Restoration Tests
# =============================================================================


def test_executable_bit_saved_in_manifest(tmp_path: pathlib.Path) -> None:
    """Executable bit should be recorded in directory manifest."""
    test_dir = tmp_path / "mydir"
    test_dir.mkdir()

    # Create executable file
    exec_file = test_dir / "script.sh"
    exec_file.write_text("#!/bin/bash\necho hello")
    exec_file.chmod(0o755)

    # Create non-executable file
    regular_file = test_dir / "data.txt"
    regular_file.write_text("data")

    _, manifest = cache.hash_directory(test_dir)

    exec_entry = next(e for e in manifest if e["relpath"] == "script.sh")
    regular_entry = next(e for e in manifest if e["relpath"] == "data.txt")

    assert exec_entry.get("isexec") is True
    assert regular_entry.get("isexec") is None or regular_entry.get("isexec") is False


def test_executable_bit_restored_with_copy_mode(tmp_path: pathlib.Path) -> None:
    """Executable bit should be restored when using COPY mode."""
    test_dir = tmp_path / "mydir"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create directory with executable file
    test_dir.mkdir()
    exec_file = test_dir / "script.sh"
    exec_file.write_text("#!/bin/bash\necho hello")
    exec_file.chmod(0o755)

    # Save to cache
    output_hash = cache.save_to_cache(test_dir, cache_dir)

    # Delete and restore with COPY mode
    cache.remove_output(test_dir)
    cache.restore_from_cache(test_dir, output_hash, cache_dir, cache.CheckoutMode.COPY)

    # Check executable bit is restored
    restored_exec = test_dir / "script.sh"
    assert restored_exec.exists()
    mode = restored_exec.stat().st_mode
    assert mode & 0o100, "Executable bit should be set"


# =============================================================================
# Uncached Incremental Output Error Tests
# =============================================================================


def test_uncached_incremental_output_raises_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """Should raise error when IncrementalOut file exists but has no cache entry."""
    from pivot import exceptions

    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "database.txt"
    db_path.write_text("uncached content\n")

    _register_incremental_stage(
        _incremental_stage_append,
        name="append_stage",
        out_path=str(db_path),
    )

    with pytest.raises(exceptions.UncachedIncrementalOutputError) as exc_info:
        executor.run(cache_dir=tmp_path / ".pivot" / "cache")

    assert "database.txt" in str(exc_info.value)
    assert "not in cache" in str(exc_info.value)


def test_uncached_incremental_output_allow_uncached_incremental_allows_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """allow_uncached_incremental=True should bypass the uncached output check."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "database.txt"
    db_path.write_text("will be overwritten\n")

    _register_incremental_stage(
        _incremental_stage_append,
        name="append_stage",
        out_path=str(db_path),
    )

    # allow_uncached_incremental=True should allow run even with uncached file
    results = executor.run(cache_dir=tmp_path / ".pivot" / "cache", allow_uncached_incremental=True)
    assert results["append_stage"]["status"] == "ran"


def test_cached_incremental_output_runs_normally(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """IncrementalOut that is properly cached should run without error."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "database.txt"
    cache_dir = tmp_path / ".pivot" / "cache"

    _register_incremental_stage(
        _incremental_stage_append,
        name="append_stage",
        out_path=str(db_path),
    )

    # First run creates and caches the output
    results1 = executor.run(cache_dir=cache_dir)
    assert results1["append_stage"]["status"] == "ran"

    # Second run should skip without error (output is cached, nothing changed)
    results2 = executor.run(cache_dir=cache_dir)
    assert results2["append_stage"]["status"] == "skipped"


def test_force_runs_incremental_stage_even_when_unchanged(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """force=True should re-run IncrementalOut stage even when nothing changed."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "database.txt"
    cache_dir = tmp_path / ".pivot" / "cache"

    _register_incremental_stage(
        _incremental_stage_append,
        name="append_stage",
        out_path=str(db_path),
    )

    # First run creates the output
    results1 = executor.run(cache_dir=cache_dir)
    assert results1["append_stage"]["status"] == "ran"
    assert db_path.read_text() == "line 1\n"

    # Second run without force should skip
    results2 = executor.run(cache_dir=cache_dir)
    assert results2["append_stage"]["status"] == "skipped"
    assert db_path.read_text() == "line 1\n"

    # Third run with force=True should run and append
    results3 = executor.run(cache_dir=cache_dir, force=True)
    assert results3["append_stage"]["status"] == "ran"
    assert results3["append_stage"]["reason"] == "forced"
    assert db_path.read_text() == "line 1\nline 2\n"


def test_force_and_allow_uncached_incremental_are_orthogonal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """force and allow_uncached_incremental are independent flags."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "database.txt"
    # Create uncached file (not from a previous run)
    db_path.write_text("uncached content\n")
    cache_dir = tmp_path / ".pivot" / "cache"

    _register_incremental_stage(
        _incremental_stage_append,
        name="append_stage",
        out_path=str(db_path),
    )

    # force=True alone should still raise error for uncached incremental
    from pivot import exceptions

    with pytest.raises(exceptions.UncachedIncrementalOutputError):
        executor.run(cache_dir=cache_dir, force=True)

    # Both flags together should work
    results = executor.run(cache_dir=cache_dir, force=True, allow_uncached_incremental=True)
    assert results["append_stage"]["status"] == "ran"


# =============================================================================
# IncrementalOut as Input Annotation Tests
# =============================================================================

# Module-level type aliases for tests (required for get_type_hints to work)
type _TestMyCache = Annotated[
    dict[str, str] | None, outputs.IncrementalOut("cache.json", loaders.JSON[dict[str, str]]())
]


class _TestCacheOutputs(TypedDict):
    existing: Annotated[
        dict[str, str], outputs.IncrementalOut("cache.json", loaders.JSON[dict[str, str]]())
    ]


def _test_stage_with_incremental_input(existing: _TestMyCache) -> _RegularStageOutputs:
    """Module-level stage function for testing IncrementalOut as input."""
    return {"output": pathlib.Path("output.txt")}


def _test_cache_stage(existing: _TestMyCache) -> _TestCacheOutputs:
    """Module-level stage function for testing IncrementalOut as both input and output."""
    if existing is None:
        existing = {}
    existing["new_key"] = "value"
    return _TestCacheOutputs(existing=existing)


def _test_stage_with_mixed_deps(
    data: Annotated[dict[str, Any], outputs.Dep("input.json", loaders.JSON[dict[str, Any]]())],
    existing: _TestMyCache,
) -> _RegularStageOutputs:
    """Module-level stage function for testing IncrementalOut mixed with Dep."""
    return {"output": pathlib.Path("output.txt")}


def test_incremental_out_recognized_as_input_annotation() -> None:
    """IncrementalOut should be recognized as a valid input annotation."""
    from pivot import stage_def

    dep_specs = stage_def.get_dep_specs_from_signature(_test_stage_with_incremental_input)
    assert "existing" in dep_specs
    assert dep_specs["existing"].path == "cache.json"
    assert dep_specs["existing"].creates_dep_edge is False


def test_incremental_input_first_run_returns_empty(tmp_path: pathlib.Path) -> None:
    """IncrementalOut as input should return empty value if file doesn't exist."""
    from pivot import stage_def

    dep_specs = stage_def.get_dep_specs_from_signature(_test_stage_with_incremental_input)
    loaded = stage_def.load_deps_from_specs(dep_specs, tmp_path)

    # JSON loader returns empty dict for first run
    assert loaded["existing"] == {}


def test_incremental_input_subsequent_run_loads_value(tmp_path: pathlib.Path) -> None:
    """IncrementalOut as input should load value if file exists."""
    from pivot import stage_def

    # Create the cache file
    cache_file = tmp_path / "cache.json"
    cache_file.write_text('{"key": "value"}')

    dep_specs = stage_def.get_dep_specs_from_signature(_test_stage_with_incremental_input)
    loaded = stage_def.load_deps_from_specs(dep_specs, tmp_path)

    assert loaded["existing"] == {"key": "value"}


def test_incremental_input_no_circular_dependency(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """IncrementalOut as input should NOT create DAG circular dependency."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    # Register the stage - should NOT raise circular dependency error
    REGISTRY.register(_test_cache_stage, name="cache_stage")

    # Build DAG - should succeed without circular dependency
    dag = REGISTRY.build_dag(validate=False)
    assert "cache_stage" in dag.nodes()


def test_incremental_input_type_alias_pattern() -> None:
    """Type alias can be shared between input parameter and return annotation."""
    from pivot import stage_def

    # Verify both input and output recognized
    dep_specs = stage_def.get_dep_specs_from_signature(_test_cache_stage)
    assert "existing" in dep_specs
    assert dep_specs["existing"].creates_dep_edge is False

    out_specs = stage_def.get_output_specs_from_return(_test_cache_stage, "_test_cache_stage")
    assert "existing" in out_specs
    assert isinstance(out_specs["existing"], outputs.IncrementalOut)


def test_incremental_input_mixed_with_regular_deps() -> None:
    """IncrementalOut input can coexist with regular Dep annotations."""
    from pivot import stage_def

    dep_specs = stage_def.get_dep_specs_from_signature(_test_stage_with_mixed_deps)

    # Regular Dep creates edge
    assert "data" in dep_specs
    assert dep_specs["data"].creates_dep_edge is True

    # IncrementalOut does not create edge
    assert "existing" in dep_specs
    assert dep_specs["existing"].creates_dep_edge is False


# =============================================================================
# IncrementalOut Validation Tests
# =============================================================================


# Module-level stage functions for validation tests (must be picklable)


class _ValidationTestWrongNameOutputs(TypedDict):
    # Output field name 'wrong_name' doesn't match input parameter name 'existing'
    wrong_name: Annotated[
        dict[str, str], outputs.IncrementalOut("cache.json", loaders.JSON[dict[str, str]]())
    ]


class _ValidationTestWrongPathOutputs(TypedDict):
    existing: Annotated[
        dict[str, str], outputs.IncrementalOut("wrong_path.json", loaders.JSON[dict[str, str]]())
    ]


class _ValidationTestWrongLoaderOutputs(TypedDict):
    existing: Annotated[
        dict[str, str], outputs.IncrementalOut("cache.json", loaders.Pickle[dict[str, str]]())
    ]


type _TestCacheA = Annotated[
    dict[str, str] | None, outputs.IncrementalOut("a.json", loaders.JSON[dict[str, str]]())
]
type _TestCacheB = Annotated[
    dict[str, str] | None, outputs.IncrementalOut("b.json", loaders.JSON[dict[str, str]]())
]


class _ValidationTestMultiOutputs(TypedDict):
    cache_a: Annotated[
        dict[str, str], outputs.IncrementalOut("a.json", loaders.JSON[dict[str, str]]())
    ]
    cache_b: Annotated[
        dict[str, str], outputs.IncrementalOut("b.json", loaders.JSON[dict[str, str]]())
    ]


def _stage_incremental_input_no_output(existing: _TestMyCache) -> _RegularStageOutputs:
    """Stage with IncrementalOut input but regular output."""
    return {"output": pathlib.Path("output.txt")}


def _stage_incremental_input_wrong_field_name(
    existing: _TestMyCache,
) -> _ValidationTestWrongNameOutputs:
    """Stage with IncrementalOut input but output field name doesn't match."""
    return _ValidationTestWrongNameOutputs(wrong_name=existing or {})


def _stage_incremental_input_wrong_path(existing: _TestMyCache) -> _ValidationTestWrongPathOutputs:
    """Stage with IncrementalOut input but output path doesn't match."""
    return _ValidationTestWrongPathOutputs(existing=existing or {})


def _stage_incremental_input_wrong_loader(
    existing: _TestMyCache,
) -> _ValidationTestWrongLoaderOutputs:
    """Stage with IncrementalOut input but output loader doesn't match."""
    return _ValidationTestWrongLoaderOutputs(existing=existing or {})


def _stage_single_output_incremental(existing: _TestMyCache) -> _TestMyCache:
    """Stage with single IncrementalOut input and matching single output."""
    if existing is None:
        existing = {}
    existing["new"] = "value"
    return existing


type _TestCacheSingleOut = Annotated[
    dict[str, str] | None, outputs.IncrementalOut("single.json", loaders.JSON[dict[str, str]]())
]


def _stage_single_output_wrong_path(
    existing: _TestMyCache,
) -> _TestCacheSingleOut:
    """Stage with single IncrementalOut but path doesn't match."""
    return existing


def _stage_multiple_incremental_inputs_single_output(
    cache_a: _TestCacheA,
    cache_b: _TestCacheB,
) -> _TestCacheA:
    """Stage with multiple IncrementalOut inputs but single output."""
    return cache_a


def _stage_multiple_incremental_matched(
    cache_a: _TestCacheA,
    cache_b: _TestCacheB,
) -> _ValidationTestMultiOutputs:
    """Stage with multiple IncrementalOut inputs matched by name."""
    return _ValidationTestMultiOutputs(
        cache_a=cache_a or {},
        cache_b=cache_b or {},
    )


def test_incremental_out_input_requires_matching_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """IncrementalOut input must have a matching IncrementalOut output."""
    from pivot import exceptions

    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(exceptions.ValidationError) as exc_info:
        REGISTRY.register(_stage_incremental_input_no_output, name="test_stage")

    # TypedDict return has outputs but none are IncrementalOut matching the input
    assert "no matching IncrementalOut output field" in str(exc_info.value)


def test_incremental_out_typeddict_name_must_match(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """For TypedDict returns, output field name must match parameter name."""
    from pivot import exceptions

    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(exceptions.ValidationError) as exc_info:
        REGISTRY.register(_stage_incremental_input_wrong_field_name, name="test_stage")

    assert "no matching IncrementalOut output field" in str(exc_info.value)
    assert "existing" in str(exc_info.value)


def test_incremental_out_input_output_path_must_match(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """IncrementalOut input and output paths must match."""
    from pivot import exceptions

    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(exceptions.ValidationError) as exc_info:
        REGISTRY.register(_stage_incremental_input_wrong_path, name="test_stage")

    assert "path" in str(exc_info.value).lower()
    assert "doesn't match" in str(exc_info.value)


def test_incremental_out_input_output_loader_must_match(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """IncrementalOut input and output loaders must match."""
    from pivot import exceptions

    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(exceptions.ValidationError) as exc_info:
        REGISTRY.register(_stage_incremental_input_wrong_loader, name="test_stage")

    assert "loader" in str(exc_info.value).lower()
    assert "doesn't match" in str(exc_info.value)


def test_incremental_out_single_output_only_one_input(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """Single-output stages can only have one IncrementalOut input."""
    from pivot import exceptions

    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(exceptions.ValidationError) as exc_info:
        REGISTRY.register(_stage_multiple_incremental_inputs_single_output, name="test_stage")

    assert "single-output stages can only have one" in str(exc_info.value)


def test_incremental_out_single_output_path_must_match(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """Single IncrementalOut input and output paths must match."""
    from pivot import exceptions

    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(exceptions.ValidationError) as exc_info:
        REGISTRY.register(_stage_single_output_wrong_path, name="test_stage")

    assert "path" in str(exc_info.value).lower()
    assert "doesn't match" in str(exc_info.value)


def test_multiple_incremental_outs_matched_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """Multiple IncrementalOut pairs work when names match."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    # Should NOT raise - names match (cache_a -> cache_a, cache_b -> cache_b)
    REGISTRY.register(_stage_multiple_incremental_matched, name="test_stage")

    info = REGISTRY.get("test_stage")
    assert len(info["outs"]) == 2


def test_incremental_out_single_output_valid(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    """Single IncrementalOut input with matching single output is valid."""
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    # Should NOT raise - input and output use same type alias (same path/loader)
    REGISTRY.register(_stage_single_output_incremental, name="test_stage")

    info = REGISTRY.get("test_stage")
    assert len(info["outs"]) == 1
    assert isinstance(info["outs"][0], outputs.IncrementalOut)
