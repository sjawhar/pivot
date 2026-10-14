# pyright: reportUnusedFunction=false
"""Tests for cross-process fingerprint determinism.

These tests verify that fingerprints are stable across separate Python process
invocations. This is critical because stages may be fingerprinted in different
processes (main process vs worker) and must produce identical results.
"""

import functools
import json
import os
import pathlib
import subprocess
import sys

import pydantic
import pytest

from pivot import fingerprint


def _schema_key(model: type[object]) -> str:
    return f"schema:{model.__module__}.{model.__qualname__}"


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

print(next(value for key, value in fingerprint.get_stage_fingerprint(stage).items() if key.startswith("schema:")))
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

print(next(value for key, value in fingerprint.get_stage_fingerprint(stage).items() if key.startswith("schema:")))
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


_SET_PARAMS_SCRIPT = """\
import json

from pivot import fingerprint, parameters
from pivot.stage_def import StageParams


class Inner(StageParams):
    keys: frozenset[str] = frozenset({"k1", "k2", "k3", "k4", "k5"})


class Params(StageParams):
    tags: frozenset[str] = frozenset({"alpha", "beta", "gamma", "delta", "eps"})
    pairs: set[tuple[str, frozenset[int]]] = {("x", frozenset({1, 2, 3})), ("y", frozenset({4, 5}))}
    inner: Inner = Inner()


def stage(params: Params = Params()) -> int:
    return len(params.tags)


print(json.dumps({
    "manifest": fingerprint.get_stage_fingerprint(stage),
    "params": parameters.get_effective_params(Params(), "s/stage"),
}, sort_keys=True))
"""


def test_set_valued_params_are_deterministic_across_hash_seeds(tmp_path: pathlib.Path) -> None:
    """Set and frozenset StageParams fields hash and record in one order, whatever the seed."""
    (tmp_path / ".pivot").mkdir()
    script = tmp_path / "set_params_stage.py"
    script.write_text(_SET_PARAMS_SCRIPT)
    results = [
        json.loads(
            subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp_path,
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        for seed in range(4)
    ]

    assert all(result["params"] == results[0]["params"] for result in results), results
    assert all(result["manifest"] == results[0]["manifest"] for result in results), results
    assert results[0]["params"]["tags"] == ["alpha", "beta", "delta", "eps", "gamma"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(frozenset({"a"}), ("a",), id="frozenset-vs-tuple"),
        pytest.param([set[str]()], ["set([])"], id="empty-set-vs-tag-string"),
        pytest.param([["set", list[str]()]], [set[str]()], id="tag-shaped-list-vs-set"),
        pytest.param({1: "x"}, {"1": "x"}, id="int-key-vs-str-key"),
        pytest.param([b"a"], ["b'a'"], id="bytes-vs-repr-string"),
    ],
)
def test_distinct_partial_arguments_have_distinct_fingerprints(
    first: object, second: object
) -> None:
    """Canonical forms keep collection types apart from look-alike values."""

    def helper(value: object) -> object:
        return value

    first_partial = functools.partial(helper, first)
    second_partial = functools.partial(helper, second)

    def first_stage() -> object:
        return first_partial()

    def second_stage() -> object:
        return second_partial()

    first_args = next(
        value
        for key, value in fingerprint.get_stage_fingerprint(first_stage).items()
        if key.endswith(".args")
    )
    second_args = next(
        value
        for key, value in fingerprint.get_stage_fingerprint(second_stage).items()
        if key.endswith(".args")
    )
    assert first_args != second_args


class _Item(pydantic.BaseModel):
    name: str = "none"


def test_model_class_partial_argument_is_hashed_as_a_class() -> None:
    """A pydantic model class is a leaf: only an instance has model_dump() to call."""

    def build(model: type[pydantic.BaseModel]) -> pydantic.BaseModel:
        return model()

    bound = functools.partial(build, _Item)

    def stage() -> pydantic.BaseModel:
        return bound()

    manifest = fingerprint.get_stage_fingerprint(stage)

    assert any(key.endswith(".args") for key in manifest)


class _OtherItem(pydantic.BaseModel):
    name: str = "other"


@pytest.mark.filterwarnings("ignore::pydantic.json_schema.PydanticJsonSchemaWarning")
def test_stage_params_field_defaulting_to_a_class_is_hashed() -> None:
    """Pydantic leaves a class default out of the schema; the defaults hash still has it."""

    class ItemParams(pydantic.BaseModel):
        model: type[pydantic.BaseModel] = _Item

    class OtherParams(pydantic.BaseModel):
        model: type[pydantic.BaseModel] = _OtherItem

    def item_stage(params: ItemParams) -> None:
        pass

    def other_stage(params: OtherParams) -> None:
        pass

    item_schema = fingerprint.get_stage_fingerprint(item_stage)[_schema_key(ItemParams)]
    other_schema = fingerprint.get_stage_fingerprint(other_stage)[_schema_key(OtherParams)]
    assert item_schema != other_schema


# --- In-process tests for Pydantic default_factory with builtins ---


def test_default_factory_list_produces_stable_hash():
    """default_factory=list produces a stable hash within the same process."""

    class Model(pydantic.BaseModel):
        items: list[str] = pydantic.Field(default_factory=list)

    def stage(params: Model) -> None:
        pass

    fp1 = fingerprint.get_stage_fingerprint(stage)
    fp2 = fingerprint.get_stage_fingerprint(stage)

    assert fp1[_schema_key(Model)] == fp2[_schema_key(Model)]


def test_default_factory_dict_produces_stable_hash():
    """default_factory=dict produces a stable hash within the same process."""

    class Model(pydantic.BaseModel):
        data: dict[str, int] = pydantic.Field(default_factory=dict)

    def stage(params: Model) -> None:
        pass

    fp1 = fingerprint.get_stage_fingerprint(stage)
    fp2 = fingerprint.get_stage_fingerprint(stage)

    assert fp1[_schema_key(Model)] == fp2[_schema_key(Model)]


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

    assert fp_list[_schema_key(ModelWithList)] != fp_dict[_schema_key(ModelWithDict)], (
        "Different builtin factories should produce different hashes"
    )
