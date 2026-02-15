# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from pivot import merkle


def test_compute_merkle_id_deterministic_hash() -> None:
    code_manifest = {"b": "2", "a": "1"}
    params = {"x": 1, "y": "test"}
    input_merkle_ids = {"b": "bb", "a": "aa"}

    result = merkle.compute_merkle_id(code_manifest, params, input_merkle_ids)

    assert result == "b2878435c62d685e"


def test_compute_merkle_id_changes_on_input() -> None:
    code_manifest = {"a": "1"}
    params = {"x": 1}
    input_merkle_ids = {"a": "aa"}

    base = merkle.compute_merkle_id(code_manifest, params, input_merkle_ids)
    changed = merkle.compute_merkle_id(code_manifest, params, {"a": "bb"})

    assert base != changed
