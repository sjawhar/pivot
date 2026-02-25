from __future__ import annotations

import re

import pytest

from pivot import run_history
from pivot.types import DepEntry, DirHash, DirManifestEntry, FileHash, StageStatus


def test_output_hash_entry_roundtrip_for_file_hash() -> None:
    entry = run_history.output_hash_to_entry("data/model.pkl", FileHash(hash="abc123def4567890"))

    assert entry == {"path": "data/model.pkl", "hash": "abc123def4567890"}
    assert run_history.entry_to_output_hash(entry) == {"hash": "abc123def4567890"}


def test_output_hash_entry_roundtrip_for_dir_hash() -> None:
    dir_hash = DirHash(
        hash="feedfacecafebeef",
        manifest=[DirManifestEntry(relpath="a.txt", hash="1111", size=1, isexec=False)],
    )

    entry = run_history.output_hash_to_entry("data/dir", dir_hash)
    restored = run_history.entry_to_output_hash(entry)

    assert "manifest" in entry
    assert restored == dir_hash


def test_entry_to_output_hash_treats_null_manifest_as_file_hash() -> None:
    entry = run_history.OutputHashEntry(path="data/x", hash="aabbccddeeff0011", manifest=None)  # pyright: ignore[reportArgumentType] - testing null manifest handling

    assert run_history.entry_to_output_hash(entry) == {"hash": "aabbccddeeff0011"}


def test_generate_run_id_format() -> None:
    run_id = run_history.generate_run_id()

    assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}$", run_id)


def test_compute_input_hash_is_stable_under_ordering() -> None:
    deps_a: list[DepEntry] = [
        DepEntry(producer="b", key=None, hash="2"),
        DepEntry(producer="a", key="x", hash="1"),
    ]
    deps_b = list(reversed(deps_a))

    hash_a = run_history.compute_input_hash(
        code_manifest={"f.py": "h1"},
        params={"lr": 0.1},
        deps=deps_a,
        out_specs=[("outs/a", True), ("metrics/m", False)],
    )
    hash_b = run_history.compute_input_hash(
        code_manifest={"f.py": "h1"},
        params={"lr": 0.1},
        deps=deps_b,
        out_specs=[("metrics/m", False), ("outs/a", True)],
    )

    assert hash_a == hash_b


def test_deserialize_run_manifest_converts_legacy_skipped_status() -> None:
    payload = (
        b'{"ended_at":"2026-01-01T00:00:01Z","execution_order":["s"],"run_id":"r1",'
        b'"started_at":"2026-01-01T00:00:00Z","stages":{"s":{"duration_ms":1,'
        b'"input_hash":null,"reason":"legacy","status":"skipped"}},'
        b'"targeted_stages":["s"]}'
    )

    manifest = run_history.deserialize_run_manifest(payload)

    assert manifest["stages"]["s"]["status"] is StageStatus.CACHED


def test_deserialize_run_manifest_missing_keys_raises() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        run_history.deserialize_run_manifest(b'{"run_id":"r1"}')


def test_deserialize_run_manifest_invalid_status_raises() -> None:
    payload = (
        b'{"ended_at":"2026-01-01T00:00:01Z","execution_order":["s"],"run_id":"r1",'
        b'"started_at":"2026-01-01T00:00:00Z","stages":{"s":{"duration_ms":1,'
        b'"input_hash":null,"reason":"bad","status":"unknown"}},'
        b'"targeted_stages":["s"]}'
    )

    with pytest.raises(ValueError, match="Expected completion status"):
        run_history.deserialize_run_manifest(payload)


def test_deserialize_run_cache_entry_validation() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        run_history.deserialize_run_cache_entry(b'{"run_id":"r1"}')

    cache_entry = run_history.deserialize_run_cache_entry(
        b'{"run_id":"r1","output_hashes":[{"path":"x","hash":"h"}]}'
    )
    assert cache_entry["run_id"] == "r1"
