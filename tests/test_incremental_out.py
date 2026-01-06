"""Integration tests for IncrementalOut feature."""

import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import IncrementalOut, cache, executor, outputs, registry

if TYPE_CHECKING:
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


class TestPrepareOutputsForExecution:
    """Tests for _prepare_outputs_for_execution helper."""

    def test_regular_out_is_deleted(self, tmp_path: pathlib.Path) -> None:
        """Regular Out should be deleted before execution."""
        output_file = tmp_path / "output.txt"
        output_file.write_text("existing content")

        stage_outs: list[outputs.BaseOut] = [outputs.Out(path=str(output_file))]
        executor._prepare_outputs_for_execution(stage_outs, None, tmp_path / "cache")

        assert not output_file.exists()

    def test_incremental_out_no_cache_creates_empty(self, tmp_path: pathlib.Path) -> None:
        """IncrementalOut with no cache should start fresh (file doesn't exist)."""
        output_file = tmp_path / "database.txt"

        stage_outs: list[outputs.BaseOut] = [outputs.IncrementalOut(path=str(output_file))]
        executor._prepare_outputs_for_execution(stage_outs, None, tmp_path / "cache")

        assert not output_file.exists()

    def test_incremental_out_restores_from_cache(self, tmp_path: pathlib.Path) -> None:
        """IncrementalOut should restore from cache before execution."""
        output_file = tmp_path / "database.txt"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a cached version
        output_file.write_text("cached content\n")
        output_hash = cache.save_to_cache(output_file, cache_dir)

        # Lock data simulating previous run
        lock_data: LockData = {
            "output_hashes": {str(output_file): output_hash},
        }

        # Prepare for execution
        stage_outs = [outputs.IncrementalOut(path=str(output_file))]
        executor._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

        # File should be restored
        assert output_file.exists()
        assert output_file.read_text() == "cached content\n"

    def test_incremental_out_restored_file_is_writable(self, tmp_path: pathlib.Path) -> None:
        """Restored IncrementalOut should be a writable copy, not symlink."""
        output_file = tmp_path / "database.txt"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a cached version
        output_file.write_text("cached content\n")
        output_hash = cache.save_to_cache(output_file, cache_dir)

        lock_data: LockData = {
            "output_hashes": {str(output_file): output_hash},
        }

        # Prepare for execution
        stage_outs = [outputs.IncrementalOut(path=str(output_file))]
        executor._prepare_outputs_for_execution(stage_outs, lock_data, cache_dir)

        # Should NOT be a symlink (should be a copy)
        assert not output_file.is_symlink()

        # Should be writable
        output_file.write_text("modified content\n")
        assert output_file.read_text() == "modified content\n"


class TestIncrementalOutDvcExport:
    """Tests for IncrementalOut DVC export."""

    def test_incremental_out_always_persist(self) -> None:
        """IncrementalOut should always export with persist: true."""
        from pivot import dvc_compat

        inc = outputs.IncrementalOut(path="database.csv")
        result = dvc_compat._build_out_entry(inc, "database.csv")
        assert result == {"database.csv": {"persist": True}}

    def test_incremental_out_with_cache_false(self) -> None:
        """IncrementalOut with cache=False should export both options."""
        from pivot import dvc_compat

        inc = outputs.IncrementalOut(path="database.csv", cache=False)
        result = dvc_compat._build_out_entry(inc, "database.csv")
        assert result == {"database.csv": {"cache": False, "persist": True}}


class TestIncrementalOutIntegration:
    """End-to-end integration tests for IncrementalOut."""

    def test_first_run_creates_output(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
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

    def test_second_run_appends_to_output(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, clean_registry: None
    ) -> None:
        """Second run should restore and append to existing output."""
        import yaml

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
        lock_file = cache_dir / "stages" / "append_stage.lock"
        lock_data = yaml.safe_load(lock_file.read_text())
        lock_data["code_manifest"] = {"self:fake": "changed_hash"}
        lock_file.write_text(yaml.dump(lock_data))

        # Delete the output file to verify restoration works
        db_path.unlink()

        # Second run - should restore from cache and append
        results = executor.run(cache_dir=cache_dir)

        assert results["append_stage"]["status"] == "ran"
        assert db_path.read_text() == "line 1\nline 2\n"
