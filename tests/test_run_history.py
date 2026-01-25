"""Tests for run history and run cache functionality."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from pivot import run_history
from pivot.storage import state
from pivot.types import DepEntry, StageStatus

if TYPE_CHECKING:
    import pathlib


# =============================================================================
# run_history module tests
# =============================================================================


def test_generate_run_id_format() -> None:
    """Run ID should be YYYYMMDD_HHMMSS_<uuid8>."""
    run_id = run_history.generate_run_id()
    # Format: 20250110_143000_abc12345
    assert re.match(r"^\d{8}_\d{6}_[a-f0-9]{8}$", run_id)


def test_generate_run_id_unique() -> None:
    """Each run ID should be unique."""
    ids = [run_history.generate_run_id() for _ in range(10)]
    assert len(set(ids)) == 10


def test_compute_input_hash_deterministic() -> None:
    """Input hash should be deterministic for same inputs."""
    code_manifest = {"self:func": "abc123"}
    params: dict[str, object] = {"lr": 0.01}
    deps = [DepEntry(path="data.csv", hash="def456")]
    out_specs = [("model.pkl", True)]

    hash1 = run_history.compute_input_hash(code_manifest, params, deps, out_specs)
    hash2 = run_history.compute_input_hash(code_manifest, params, deps, out_specs)
    assert hash1 == hash2


def test_compute_input_hash_different_for_different_inputs() -> None:
    """Input hash should differ when inputs change."""
    code_manifest = {"self:func": "abc123"}
    params: dict[str, object] = {"lr": 0.01}
    deps = [DepEntry(path="data.csv", hash="def456")]
    out_specs = [("model.pkl", True)]

    hash1 = run_history.compute_input_hash(code_manifest, params, deps, out_specs)

    # Change code manifest
    hash2 = run_history.compute_input_hash({"self:func": "changed"}, params, deps, out_specs)
    assert hash1 != hash2

    # Change params
    hash3 = run_history.compute_input_hash(code_manifest, {"lr": 0.02}, deps, out_specs)
    assert hash1 != hash3

    # Change deps
    hash4 = run_history.compute_input_hash(
        code_manifest,
        params,
        [DepEntry(path="data.csv", hash="changed")],
        out_specs,
    )
    assert hash1 != hash4

    # Change output cache flag (Out -> Metric conversion)
    hash5 = run_history.compute_input_hash(code_manifest, params, deps, [("model.pkl", False)])
    assert hash1 != hash5, "Cache flag change should produce different hash"


def test_compute_input_hash_order_independent_for_deps() -> None:
    """Input hash should be order-independent for deps and output specs."""
    deps_a = [
        DepEntry(path="a.csv", hash="aaa"),
        DepEntry(path="b.csv", hash="bbb"),
    ]
    deps_b = [
        DepEntry(path="b.csv", hash="bbb"),
        DepEntry(path="a.csv", hash="aaa"),
    ]

    hash_a = run_history.compute_input_hash({}, {}, deps_a, [])
    hash_b = run_history.compute_input_hash({}, {}, deps_b, [])
    assert hash_a == hash_b

    # Output specs order should also be independent
    out_specs_a = [("a.txt", True), ("b.txt", False)]
    out_specs_b = [("b.txt", False), ("a.txt", True)]
    hash_c = run_history.compute_input_hash({}, {}, [], out_specs_a)
    hash_d = run_history.compute_input_hash({}, {}, [], out_specs_b)
    assert hash_c == hash_d


def test_serialize_deserialize_run_manifest() -> None:
    """RunManifest should serialize and deserialize correctly."""
    manifest = run_history.RunManifest(
        run_id="20250110_143000_abc12345",
        started_at="2025-01-10T14:30:00+00:00",
        ended_at="2025-01-10T14:35:00+00:00",
        targeted_stages=["train", "eval"],
        execution_order=["train", "eval"],
        stages={
            "train": run_history.StageRunRecord(
                input_hash="abc123",
                status=StageStatus.RAN,
                reason="Code changed",
                duration_ms=5000,
            ),
            "eval": run_history.StageRunRecord(
                input_hash="def456",
                status=StageStatus.SKIPPED,
                reason="unchanged",
                duration_ms=0,
            ),
        },
    )

    serialized = run_history.serialize_to_bytes(manifest)
    deserialized = run_history.deserialize_run_manifest(serialized)

    assert deserialized["run_id"] == manifest["run_id"]
    assert deserialized["started_at"] == manifest["started_at"]
    assert deserialized["ended_at"] == manifest["ended_at"]
    assert deserialized["targeted_stages"] == manifest["targeted_stages"]
    assert deserialized["execution_order"] == manifest["execution_order"]
    assert deserialized["stages"]["train"]["status"] == StageStatus.RAN
    assert deserialized["stages"]["eval"]["status"] == StageStatus.SKIPPED


def test_serialize_deserialize_run_cache_entry() -> None:
    """RunCacheEntry should serialize and deserialize correctly."""
    entry = run_history.RunCacheEntry(
        run_id="20250110_143000_abc12345",
        output_hashes=[
            run_history.OutputHashEntry(path="model.pkl", hash="xyz789"),
            run_history.OutputHashEntry(path="metrics.json", hash="uvw012"),
        ],
    )

    serialized = run_history.serialize_to_bytes(entry)
    deserialized = run_history.deserialize_run_cache_entry(serialized)

    assert deserialized["run_id"] == entry["run_id"]
    assert len(deserialized["output_hashes"]) == 2
    assert deserialized["output_hashes"][0]["path"] == "model.pkl"


def test_deserialize_run_manifest_missing_keys() -> None:
    """Deserialization should raise ValueError for missing required keys."""
    incomplete = json.dumps({"run_id": "test", "started_at": ""}).encode()
    with pytest.raises(ValueError, match="missing keys"):
        run_history.deserialize_run_manifest(incomplete)


def test_deserialize_run_manifest_invalid_status() -> None:
    """Deserialization should raise ValueError for invalid status."""
    invalid_data = {
        "run_id": "test",
        "started_at": "",
        "ended_at": "",
        "targeted_stages": [],
        "execution_order": ["my_stage"],
        "stages": {
            "my_stage": {
                "input_hash": "abc",
                "status": "invalid_status",
                "reason": "test",
                "duration_ms": 0,
            }
        },
    }
    with pytest.raises(ValueError, match="Invalid status 'invalid_status' for stage 'my_stage'"):
        run_history.deserialize_run_manifest(json.dumps(invalid_data).encode())


def test_deserialize_run_cache_entry_missing_keys() -> None:
    """Deserialization should raise ValueError for missing required keys."""
    incomplete = json.dumps({"run_id": "test"}).encode()
    with pytest.raises(ValueError, match="missing keys"):
        run_history.deserialize_run_cache_entry(incomplete)


# =============================================================================
# StateDB run history tests
# =============================================================================


def test_state_db_write_read_run(tmp_path: pathlib.Path) -> None:
    """StateDB should write and read run manifests."""
    with state.StateDB(tmp_path / "state.db") as db:
        manifest = run_history.RunManifest(
            run_id="20250110_143000_abc12345",
            started_at="2025-01-10T14:30:00+00:00",
            ended_at="2025-01-10T14:35:00+00:00",
            targeted_stages=["stage1"],
            execution_order=["stage1"],
            stages={
                "stage1": run_history.StageRunRecord(
                    input_hash="hash1",
                    status=StageStatus.RAN,
                    reason="test",
                    duration_ms=100,
                )
            },
        )
        db.write_run(manifest)

        result = db.read_run("20250110_143000_abc12345")
        assert result is not None
        assert result["run_id"] == manifest["run_id"]
        assert result["stages"]["stage1"]["status"] == StageStatus.RAN


def test_state_db_read_nonexistent_run(tmp_path: pathlib.Path) -> None:
    """StateDB should return None for nonexistent run."""
    with state.StateDB(tmp_path / "state.db") as db:
        result = db.read_run("nonexistent")
        assert result is None


def test_state_db_list_runs_ordering(tmp_path: pathlib.Path) -> None:
    """StateDB should list runs in reverse chronological order."""
    with state.StateDB(tmp_path / "state.db") as db:
        # Write runs with different timestamps
        for i in range(3):
            manifest = run_history.RunManifest(
                run_id=f"2025011{i}_143000_abc12345",
                started_at="2025-01-10T14:30:00+00:00",
                ended_at="2025-01-10T14:35:00+00:00",
                targeted_stages=[],
                execution_order=[],
                stages={},
            )
            db.write_run(manifest)

        runs = db.list_runs(limit=10)
        assert len(runs) == 3
        # Most recent first (highest timestamp)
        assert runs[0]["run_id"] == "20250112_143000_abc12345"
        assert runs[1]["run_id"] == "20250111_143000_abc12345"
        assert runs[2]["run_id"] == "20250110_143000_abc12345"


def test_state_db_list_runs_limit(tmp_path: pathlib.Path) -> None:
    """StateDB should respect limit parameter."""
    with state.StateDB(tmp_path / "state.db") as db:
        for i in range(5):
            manifest = run_history.RunManifest(
                run_id=f"2025011{i}_143000_abc12345",
                started_at="",
                ended_at="",
                targeted_stages=[],
                execution_order=[],
                stages={},
            )
            db.write_run(manifest)

        runs = db.list_runs(limit=3)
        assert len(runs) == 3


def test_state_db_prune_runs(tmp_path: pathlib.Path) -> None:
    """StateDB should prune old runs beyond retention limit."""
    with state.StateDB(tmp_path / "state.db") as db:
        # Write 5 runs
        for i in range(5):
            manifest = run_history.RunManifest(
                run_id=f"2025011{i}_143000_abc12345",
                started_at="",
                ended_at="",
                targeted_stages=[],
                execution_order=[],
                stages={},
            )
            db.write_run(manifest)

        # Prune to keep only 2
        deleted = db.prune_runs(retention=2)
        assert deleted == 3

        # Should have 2 remaining (most recent)
        runs = db.list_runs(limit=10)
        assert len(runs) == 2
        assert runs[0]["run_id"] == "20250114_143000_abc12345"
        assert runs[1]["run_id"] == "20250113_143000_abc12345"


def test_state_db_prune_runs_no_op_when_under_limit(tmp_path: pathlib.Path) -> None:
    """StateDB prune should do nothing when under limit."""
    with state.StateDB(tmp_path / "state.db") as db:
        manifest = run_history.RunManifest(
            run_id="20250110_143000_abc12345",
            started_at="",
            ended_at="",
            targeted_stages=[],
            execution_order=[],
            stages={},
        )
        db.write_run(manifest)

        deleted = db.prune_runs(retention=10)
        assert deleted == 0

        runs = db.list_runs(limit=10)
        assert len(runs) == 1


# =============================================================================
# StateDB run cache tests
# =============================================================================


def test_state_db_run_cache_write_lookup(tmp_path: pathlib.Path) -> None:
    """StateDB should write and lookup run cache entries."""
    with state.StateDB(tmp_path / "state.db") as db:
        entry = run_history.RunCacheEntry(
            run_id="20250110_143000_abc12345",
            output_hashes=[run_history.OutputHashEntry(path="out.txt", hash="abc")],
        )
        db.write_run_cache("my_stage", "input_hash_123", entry)

        result = db.lookup_run_cache("my_stage", "input_hash_123")
        assert result is not None
        assert result["run_id"] == entry["run_id"]
        assert result["output_hashes"][0]["hash"] == "abc"


def test_state_db_run_cache_lookup_nonexistent(tmp_path: pathlib.Path) -> None:
    """StateDB should return None for nonexistent run cache entry."""
    with state.StateDB(tmp_path / "state.db") as db:
        result = db.lookup_run_cache("nonexistent", "nonexistent")
        assert result is None


def test_state_db_run_cache_different_stages(tmp_path: pathlib.Path) -> None:
    """Run cache entries for different stages should be independent."""
    with state.StateDB(tmp_path / "state.db") as db:
        entry1 = run_history.RunCacheEntry(
            run_id="run1",
            output_hashes=[run_history.OutputHashEntry(path="out1.txt", hash="hash1")],
        )
        entry2 = run_history.RunCacheEntry(
            run_id="run2",
            output_hashes=[run_history.OutputHashEntry(path="out2.txt", hash="hash2")],
        )

        # Same input hash, different stages
        db.write_run_cache("stage1", "same_input_hash", entry1)
        db.write_run_cache("stage2", "same_input_hash", entry2)

        result1 = db.lookup_run_cache("stage1", "same_input_hash")
        result2 = db.lookup_run_cache("stage2", "same_input_hash")

        assert result1 is not None
        assert result2 is not None
        assert result1["run_id"] == "run1"
        assert result2["run_id"] == "run2"


def test_state_db_run_cache_overwrite(tmp_path: pathlib.Path) -> None:
    """Writing to same run cache key should overwrite."""
    with state.StateDB(tmp_path / "state.db") as db:
        entry1 = run_history.RunCacheEntry(run_id="run1", output_hashes=[])
        entry2 = run_history.RunCacheEntry(run_id="run2", output_hashes=[])

        db.write_run_cache("stage", "input_hash", entry1)
        db.write_run_cache("stage", "input_hash", entry2)

        result = db.lookup_run_cache("stage", "input_hash")
        assert result is not None
        assert result["run_id"] == "run2"


def test_state_db_prune_run_cache(tmp_path: pathlib.Path) -> None:
    """Run cache pruning should remove entries referencing invalid run_ids."""
    with state.StateDB(tmp_path / "state.db") as db:
        # Create cache entries with different run_ids
        entry1 = run_history.RunCacheEntry(run_id="valid_run", output_hashes=[])
        entry2 = run_history.RunCacheEntry(run_id="orphan_run", output_hashes=[])

        db.write_run_cache("stage1", "hash1", entry1)
        db.write_run_cache("stage2", "hash2", entry2)

        # Prune with only valid_run as valid
        deleted = db.prune_run_cache({"valid_run"})
        assert deleted == 1

        # valid_run entry should still exist
        assert db.lookup_run_cache("stage1", "hash1") is not None
        # orphan_run entry should be deleted
        assert db.lookup_run_cache("stage2", "hash2") is None


def test_state_db_prune_runs_also_prunes_cache(tmp_path: pathlib.Path) -> None:
    """Pruning runs should also prune run cache entries referencing deleted runs."""
    with state.StateDB(tmp_path / "state.db") as db:
        # Create 3 runs
        for i in range(3):
            manifest = run_history.RunManifest(
                run_id=f"2025011{i}_143000_abc12345",
                started_at="",
                ended_at="",
                targeted_stages=[],
                execution_order=[],
                stages={},
            )
            db.write_run(manifest)
            # Create a cache entry for each run
            entry = run_history.RunCacheEntry(
                run_id=f"2025011{i}_143000_abc12345",
                output_hashes=[],
            )
            db.write_run_cache(f"stage{i}", f"hash{i}", entry)

        # Prune to keep only 1 run (most recent: 20250112_143000_abc12345)
        deleted = db.prune_runs(retention=1)
        assert deleted == 2

        # Only the cache entry for the remaining run should exist
        assert db.lookup_run_cache("stage2", "hash2") is not None  # kept
        assert db.lookup_run_cache("stage0", "hash0") is None  # pruned
        assert db.lookup_run_cache("stage1", "hash1") is None  # pruned
