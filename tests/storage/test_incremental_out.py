import pathlib

import pytest

from pivot import IncrementalOut, executor, outputs, registry
from pivot.executor import worker
from pivot.storage import cache, lock
from pivot.types import LockData


# Module-level stage functions for testing (must be picklable)
def _incremental_stage_append() -> None:
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


def _regular_stage_create() -> None:
    """Stage that creates a regular output."""
    pathlib.Path("output.txt").write_text("created\n")


# =============================================================================
# Prepare Outputs for Execution Tests
# =============================================================================


def test_prepare_outputs_regular_out_is_deleted(tmp_path: pathlib.Path) -> None:
    """Regular Out should be deleted before execution."""
    output_file = tmp_path / "output.txt"
    output_file.write_text("existing content")

    stage_outs: list[outputs.BaseOut] = [outputs.Out(path=str(output_file))]
    worker._prepare_outputs_for_execution(stage_outs, None, tmp_path / "cache")

    assert not output_file.exists()


def test_prepare_outputs_incremental_no_cache_creates_empty(tmp_path: pathlib.Path) -> None:
    """IncrementalOut with no cache should start fresh (file doesn't exist)."""
    output_file = tmp_path / "database.txt"

    stage_outs: list[outputs.BaseOut] = [outputs.IncrementalOut(path=str(output_file))]
    worker._prepare_outputs_for_execution(stage_outs, None, tmp_path / "cache")

    assert not output_file.exists()


def test_prepare_outputs_incremental_restores_from_cache(tmp_path: pathlib.Path) -> None:
    """IncrementalOut should restore from cache before execution."""
    output_file = tmp_path / "database.txt"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create a cached version
    output_file.write_text("cached content\n")
    output_hash = cache.save_to_cache(output_file, cache_dir)

    # Lock data simulating previous run
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={str(output_file): output_hash},
        dep_generations={},
    )

    # Prepare for execution
    stage_outs = [outputs.IncrementalOut(path=str(output_file))]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # File should be restored
    assert output_file.exists()
    assert output_file.read_text() == "cached content\n"


def test_prepare_outputs_incremental_restored_file_is_writable(tmp_path: pathlib.Path) -> None:
    """Restored IncrementalOut should be a writable copy, not symlink."""
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
        output_hashes={str(output_file): output_hash},
        dep_generations={},
    )

    # Prepare for execution
    stage_outs = [outputs.IncrementalOut(path=str(output_file))]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # Should NOT be a symlink (should be a copy)
    assert not output_file.is_symlink()

    # Should be writable
    output_file.write_text("modified content\n")
    assert output_file.read_text() == "modified content\n"


# =============================================================================
# IncrementalOut DVC Export Tests
# =============================================================================


def test_dvc_export_incremental_out_always_persist() -> None:
    """IncrementalOut should always export with persist: true."""
    from pivot import dvc_compat

    inc = outputs.IncrementalOut(path="database.csv")
    result = dvc_compat._build_out_entry(inc, "database.csv")
    assert result == {"database.csv": {"persist": True}}


def test_dvc_export_incremental_out_with_cache_false() -> None:
    """IncrementalOut with cache=False should export both options."""
    from pivot import dvc_compat

    inc = outputs.IncrementalOut(path="database.csv", cache=False)
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

    registry.REGISTRY.register(
        _incremental_stage_append,
        name="append_stage",
        deps=[],
        outs=[IncrementalOut(path=str(db_path))],
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

    registry.REGISTRY.register(
        _incremental_stage_append,
        name="append_stage",
        deps=[],
        outs=[IncrementalOut(path=str(db_path))],
    )

    # First run
    executor.run(cache_dir=cache_dir)
    assert db_path.read_text() == "line 1\n"

    # Simulate code change by modifying the lock file's code_manifest
    # Keep output_hashes so we can restore
    stage_lock = lock.StageLock("append_stage", cache_dir)
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


def test_incremental_out_restores_directory(tmp_path: pathlib.Path) -> None:
    """IncrementalOut should restore directory from cache with COPY mode."""
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

    # Simulate lock data from previous run
    lock_data = LockData(
        code_manifest={},
        params={},
        dep_hashes={},
        output_hashes={str(output_dir): output_hash},
        dep_generations={},
    )

    # Delete the output
    cache.remove_output(output_dir)
    assert not output_dir.exists()

    # Prepare for execution (restore with COPY mode)
    stage_outs = [outputs.IncrementalOut(path=str(output_dir))]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # Directory should be restored
    assert output_dir.exists()
    assert (output_dir / "file1.txt").read_text() == "content1"
    assert (output_dir / "subdir" / "file2.txt").read_text() == "content2"


def test_incremental_out_directory_is_writable(tmp_path: pathlib.Path) -> None:
    """Restored directory should allow creating new files."""
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
        output_hashes={str(output_dir): output_hash},
        dep_generations={},
    )

    # Delete and restore
    cache.remove_output(output_dir)
    stage_outs = [outputs.IncrementalOut(path=str(output_dir))]
    worker._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

    # Should be able to write new files
    new_file = output_dir / "new_file.txt"
    new_file.write_text("new content")
    assert new_file.read_text() == "new content"

    # Should be able to modify existing files
    (output_dir / "existing.txt").write_text("modified")
    assert (output_dir / "existing.txt").read_text() == "modified"


def test_incremental_out_directory_subdirs_writable(tmp_path: pathlib.Path) -> None:
    """Restored subdirectories should allow creating new files."""
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
        output_hashes={str(output_dir): output_hash},
        dep_generations={},
    )

    # Delete and restore
    cache.remove_output(output_dir)
    stage_outs = [outputs.IncrementalOut(path=str(output_dir))]
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

    registry.REGISTRY.register(
        _incremental_stage_append,
        name="append_stage",
        deps=[],
        outs=[IncrementalOut(path=str(db_path))],
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

    registry.REGISTRY.register(
        _incremental_stage_append,
        name="append_stage",
        deps=[],
        outs=[IncrementalOut(path=str(db_path))],
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

    registry.REGISTRY.register(
        _incremental_stage_append,
        name="append_stage",
        deps=[],
        outs=[IncrementalOut(path=str(db_path))],
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

    registry.REGISTRY.register(
        _incremental_stage_append,
        name="append_stage",
        deps=[],
        outs=[IncrementalOut(path=str(db_path))],
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

    registry.REGISTRY.register(
        _incremental_stage_append,
        name="append_stage",
        deps=[],
        outs=[IncrementalOut(path=str(db_path))],
    )

    # force=True alone should still raise error for uncached incremental
    from pivot import exceptions

    with pytest.raises(exceptions.UncachedIncrementalOutputError):
        executor.run(cache_dir=cache_dir, force=True)

    # Both flags together should work
    results = executor.run(cache_dir=cache_dir, force=True, allow_uncached_incremental=True)
    assert results["append_stage"]["status"] == "ran"
