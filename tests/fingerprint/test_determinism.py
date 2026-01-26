# pyright: reportUnusedFunction=false
"""Tests for cross-process fingerprint determinism.

These tests verify that fingerprints are stable across separate Python process
invocations. This is critical because stages may be fingerprinted in different
processes (main process vs worker) and must produce identical results.
"""

import subprocess
import sys

import pydantic
import pytest

from pivot import fingerprint

# --- Unit tests for builtin handling ---


def test_builtin_type_deterministic():
    """Builtin types like list, dict, set produce stable hashes."""
    hash1 = fingerprint.hash_function_ast(list)
    hash2 = fingerprint.hash_function_ast(list)
    assert hash1 == hash2, "Builtin type hash should be stable"


def test_builtin_types_have_different_hashes():
    """Different builtin types produce different hashes."""
    hashes = {
        "list": fingerprint.hash_function_ast(list),
        "dict": fingerprint.hash_function_ast(dict),
        "set": fingerprint.hash_function_ast(set),
        "tuple": fingerprint.hash_function_ast(tuple),
        "frozenset": fingerprint.hash_function_ast(frozenset),
        "str": fingerprint.hash_function_ast(str),
        "int": fingerprint.hash_function_ast(int),
        "float": fingerprint.hash_function_ast(float),
        "bool": fingerprint.hash_function_ast(bool),
        "bytes": fingerprint.hash_function_ast(bytes),
    }
    # All hashes should be unique
    assert len(set(hashes.values())) == len(hashes), (
        f"All builtin types should have unique hashes: {hashes}"
    )


def test_builtin_hash_format():
    """Builtin type hashes are valid 16-char hex strings."""
    hash_val = fingerprint.hash_function_ast(list)
    assert len(hash_val) == 16, f"Expected 16-char hash, got {len(hash_val)}"
    assert all(c in "0123456789abcdef" for c in hash_val)


# --- Module attribute fingerprinting tests ---


def test_stdlib_module_attributes_not_in_fingerprint():
    """Non-user-code module attributes (numpy, etc.) should not be fingerprinted.

    Uses subprocess to import numpy at module level (how real stages work).
    """
    pytest.importorskip("numpy")
    script = """\
import numpy as np
from pivot import fingerprint

def stage_using_numpy():
    arr = np.linspace(0, 1, 10)
    combined = np.c_[arr, arr]
    return combined

manifest = fingerprint.get_stage_fingerprint(stage_using_numpy)
mod_keys = [k for k in manifest if k.startswith("mod:")]
print(f"mod_keys: {mod_keys}")
if mod_keys:
    print(f"ERROR: Unexpected stdlib module attributes in fingerprint: {mod_keys}")
    exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Test failed:\n{result.stdout}\n{result.stderr}"


# --- Cross-process determinism tests ---


@pytest.mark.slow
def test_builtin_default_factory_deterministic_across_processes():
    """Fingerprints with default_factory=list are stable across processes."""
    script = """
import pydantic
from pivot import fingerprint

class Model(pydantic.BaseModel):
    items: list[str] = pydantic.Field(default_factory=list)

def stage(params: Model) -> None: pass

print(fingerprint.get_stage_fingerprint(stage)["pydantic:Model.items"])
"""
    results = list[str]()
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(result.stdout.strip())

    assert results[0] == results[1], f"Hashes differ across processes: {results}"


@pytest.mark.slow
def test_builtin_dict_factory_deterministic_across_processes():
    """Fingerprints with default_factory=dict are stable across processes."""
    script = """
import pydantic
from pivot import fingerprint

class Model(pydantic.BaseModel):
    data: dict[str, int] = pydantic.Field(default_factory=dict)

def stage(params: Model) -> None: pass

print(fingerprint.get_stage_fingerprint(stage)["pydantic:Model.data"])
"""
    results = list[str]()
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(result.stdout.strip())

    assert results[0] == results[1], f"Hashes differ across processes: {results}"


@pytest.mark.slow
def test_hash_function_ast_builtin_deterministic_across_processes():
    """hash_function_ast(list) is stable across processes."""
    script = """
from pivot import fingerprint
print(fingerprint.hash_function_ast(list))
"""
    results = list[str]()
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(result.stdout.strip())

    assert results[0] == results[1], f"Hashes differ across processes: {results}"


# --- In-process tests for Pydantic default_factory with builtins ---


def test_default_factory_list_produces_stable_hash():
    """default_factory=list produces a stable hash within the same process."""

    class Model(pydantic.BaseModel):
        items: list[str] = pydantic.Field(default_factory=list)

    def stage(params: Model) -> None:
        pass

    fp1 = fingerprint.get_stage_fingerprint(stage)
    fp2 = fingerprint.get_stage_fingerprint(stage)

    assert fp1["pydantic:Model.items"] == fp2["pydantic:Model.items"]


def test_default_factory_dict_produces_stable_hash():
    """default_factory=dict produces a stable hash within the same process."""

    class Model(pydantic.BaseModel):
        data: dict[str, int] = pydantic.Field(default_factory=dict)

    def stage(params: Model) -> None:
        pass

    fp1 = fingerprint.get_stage_fingerprint(stage)
    fp2 = fingerprint.get_stage_fingerprint(stage)

    assert fp1["pydantic:Model.data"] == fp2["pydantic:Model.data"]


def test_different_builtin_factories_have_different_hashes():
    """default_factory=list vs default_factory=dict produce different hashes."""

    class ModelWithList(pydantic.BaseModel):
        items: list[str] = pydantic.Field(default_factory=list)

    class ModelWithDict(pydantic.BaseModel):
        items: dict[str, str] = pydantic.Field(default_factory=dict)

    def stage_list(params: ModelWithList) -> None:
        pass

    def stage_dict(params: ModelWithDict) -> None:
        pass

    fp_list = fingerprint.get_stage_fingerprint(stage_list)
    fp_dict = fingerprint.get_stage_fingerprint(stage_dict)

    assert fp_list["pydantic:ModelWithList.items"] != fp_dict["pydantic:ModelWithDict.items"], (
        "Different builtin factories should produce different hashes"
    )
