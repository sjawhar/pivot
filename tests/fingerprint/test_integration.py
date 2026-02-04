import importlib
import pathlib
import sys
import types
from typing import TYPE_CHECKING

import pytest

from pivot import fingerprint

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def module_dir(tmp_path: pathlib.Path) -> "Generator[pathlib.Path]":
    """Create a temporary directory for test modules and add to sys.path."""
    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))
    # Clean up any modules we imported from this directory
    to_remove = [name for name in sys.modules if name.startswith("test_mod_")]
    for name in to_remove:
        del sys.modules[name]


def _import_module(name: str) -> types.ModuleType:
    """Import a module by name, handling cache invalidation."""
    importlib.invalidate_caches()
    return importlib.import_module(name)


def _reimport_module(name: str) -> types.ModuleType:
    """Re-import a module after clearing from sys.modules."""
    if name in sys.modules:
        del sys.modules[name]
    return _import_module(name)


def test_module_attr_change_detected(module_dir: pathlib.Path) -> None:
    """Changing helper accessed via module.attr changes fingerprint."""
    # Write helper module V1
    helpers_py = module_dir / "test_mod_helpers_v1.py"
    helpers_py.write_text("""
def process(x):
    return x * 2
""")

    # Write stage module that uses Google-style import
    stage_py = module_dir / "test_mod_stage_v1.py"
    stage_py.write_text("""
import test_mod_helpers_v1 as helpers

def run_stage():
    return helpers.process(10)
""")

    # Import and get fingerprint V1
    stage_mod = _import_module("test_mod_stage_v1")
    fp1 = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Verify helper is captured with a hash (not "callable")
    helper_key = "mod:helpers.process"
    assert helper_key in fp1, f"Should capture helper.process, got: {fp1.keys()}"
    hash1 = fp1[helper_key]
    assert len(hash1) == 16, f"Should be 16-char hash, got: {hash1}"
    assert all(c in "0123456789abcdef" for c in hash1), f"Should be hex, got: {hash1}"

    # Modify helper function
    helpers_py.write_text("""
def process(x):
    return x * 3  # CHANGED!
""")

    # Force re-import
    _reimport_module("test_mod_helpers_v1")
    stage_mod_v2 = _reimport_module("test_mod_stage_v1")
    fp2 = fingerprint.get_stage_fingerprint(stage_mod_v2.run_stage)

    # Fingerprint MUST be different
    hash2 = fp2[helper_key]
    assert hash1 != hash2, f"Fingerprint must change when helper changes: {hash1} vs {hash2}"


def test_unchanged_code_same_fingerprint(module_dir: pathlib.Path) -> None:
    """Same code produces same fingerprint (stability check)."""
    helpers_py = module_dir / "test_mod_helpers_v2.py"
    helpers_py.write_text("""
def process(x):
    return x * 2
""")

    stage_py = module_dir / "test_mod_stage_v2.py"
    stage_py.write_text("""
import test_mod_helpers_v2 as helpers

def run_stage():
    return helpers.process(10)
""")

    stage_mod = _import_module("test_mod_stage_v2")
    fp1 = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Get fingerprint again without changes
    fp2 = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    assert fp1 == fp2, "Same code must produce same fingerprint"


def test_transitive_change_detected(module_dir: pathlib.Path) -> None:
    """Changing transitive dependency changes fingerprint."""
    # Write leaf module
    leaf_py = module_dir / "test_mod_leaf_v3.py"
    leaf_py.write_text("""
def leaf_func(x):
    return x + 1
""")

    # Write helper that uses leaf
    helpers_py = module_dir / "test_mod_helpers_v3.py"
    helpers_py.write_text("""
import test_mod_leaf_v3 as leaf

def process(x):
    return leaf.leaf_func(x) * 2
""")

    # Write stage that uses helper
    stage_py = module_dir / "test_mod_stage_v3.py"
    stage_py.write_text("""
import test_mod_helpers_v3 as helpers

def run_stage():
    return helpers.process(10)
""")

    stage_mod = _import_module("test_mod_stage_v3")
    fp1 = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Verify transitive dep is captured
    assert "mod:leaf.leaf_func" in fp1, f"Should capture transitive dep, got: {fp1.keys()}"

    # Modify leaf function
    leaf_py.write_text("""
def leaf_func(x):
    return x + 100  # CHANGED!
""")

    # Force re-import of all modules
    for mod_name in ["test_mod_leaf_v3", "test_mod_helpers_v3", "test_mod_stage_v3"]:
        _reimport_module(mod_name)

    stage_mod_v2 = _import_module("test_mod_stage_v3")
    fp2 = fingerprint.get_stage_fingerprint(stage_mod_v2.run_stage)

    # Transitive fingerprint must change
    assert fp1["mod:leaf.leaf_func"] != fp2["mod:leaf.leaf_func"], (
        "Transitive dep fingerprint must change"
    )


def test_stdlib_module_attrs_not_tracked(module_dir: pathlib.Path) -> None:
    """Stdlib module attributes are NOT tracked in fingerprint."""
    stage_py = module_dir / "test_mod_stage_v4.py"
    stage_py.write_text("""
import json as json_mod

def run_stage():
    return json_mod.dumps({"key": "value"})
""")

    stage_mod = _import_module("test_mod_stage_v4")
    fp = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # json is stdlib - should NOT be in fingerprint
    assert "mod:json_mod.dumps" not in fp, (
        f"Stdlib module attrs should not be tracked, got: {list(fp.keys())}"
    )


def test_direct_import_change_detected(module_dir: pathlib.Path) -> None:
    """Changing directly imported function changes fingerprint."""
    # Write helper module V1
    helpers_py = module_dir / "test_mod_helpers_v5.py"
    helpers_py.write_text("""
def helper_func(x):
    return x * 2
""")

    # Write stage that uses direct import
    stage_py = module_dir / "test_mod_stage_v5.py"
    stage_py.write_text("""
from test_mod_helpers_v5 import helper_func

def run_stage():
    return helper_func(10)
""")

    stage_mod = _import_module("test_mod_stage_v5")
    fp1 = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Verify helper is captured
    assert "func:helper_func" in fp1, f"Should capture helper_func, got: {fp1.keys()}"
    hash1 = fp1["func:helper_func"]

    # Modify helper
    helpers_py.write_text("""
def helper_func(x):
    return x * 3  # CHANGED!
""")

    # Force re-import
    _reimport_module("test_mod_helpers_v5")
    stage_mod_v2 = _reimport_module("test_mod_stage_v5")
    fp2 = fingerprint.get_stage_fingerprint(stage_mod_v2.run_stage)

    # Fingerprint must change
    hash2 = fp2["func:helper_func"]
    assert hash1 != hash2, f"Direct import fingerprint must change: {hash1} vs {hash2}"


def test_constant_via_module_attr_captured(module_dir: pathlib.Path) -> None:
    """Constants accessed via module.attr are captured."""
    helpers_py = module_dir / "test_mod_helpers_v6.py"
    helpers_py.write_text("""
THRESHOLD = 0.5

def process(x):
    return x > THRESHOLD
""")

    stage_py = module_dir / "test_mod_stage_v6.py"
    stage_py.write_text("""
import test_mod_helpers_v6 as helpers

def run_stage():
    return helpers.process(0.6) and helpers.THRESHOLD < 1.0
""")

    stage_mod = _import_module("test_mod_stage_v6")
    fp = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Constant should be captured with repr value
    assert "mod:helpers.THRESHOLD" in fp, f"Should capture constant, got: {fp.keys()}"
    assert fp["mod:helpers.THRESHOLD"] == "0.5", (
        f"Constant value should be repr, got: {fp['mod:helpers.THRESHOLD']}"
    )


def test_both_import_styles_in_same_stage(module_dir: pathlib.Path) -> None:
    """Stage using both direct and Google-style imports works."""
    helpers_py = module_dir / "test_mod_helpers_v7.py"
    helpers_py.write_text("""
def func_a(x):
    return x + 1

def func_b(x):
    return x * 2
""")

    stage_py = module_dir / "test_mod_stage_v7.py"
    stage_py.write_text("""
from test_mod_helpers_v7 import func_a
import test_mod_helpers_v7 as helpers

def run_stage():
    return func_a(10) + helpers.func_b(20)
""")

    stage_mod = _import_module("test_mod_stage_v7")
    fp = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Both should be captured
    assert "func:func_a" in fp, "Should capture direct import"
    assert "mod:helpers.func_b" in fp, "Should capture module attr"

    # Both should be hashes (not "callable")
    assert len(fp["func:func_a"]) == 16, "Direct import should be hashed"
    assert len(fp["mod:helpers.func_b"]) == 16, "Module attr should be hashed"


def test_unsupported_module_attr_type_raises_error(module_dir: pathlib.Path) -> None:
    """Unsupported types (custom objects, non-primitive collections) in module attrs raise TypeError."""
    helpers_py = module_dir / "test_mod_helpers_v8.py"
    helpers_py.write_text("""
# This is an unsupported type - a list containing a custom object
class Config:
    def __init__(self, value: int) -> None:
        self.value = value

MY_CONFIGS = [Config(1), Config(2)]

def process(x: int) -> bool:
    return any(c.value == x for c in MY_CONFIGS)
""")

    stage_py = module_dir / "test_mod_stage_v8.py"
    stage_py.write_text("""
import test_mod_helpers_v8 as helpers

def run_stage():
    # Uses the list via module attribute access
    return helpers.process(1) and len(helpers.MY_CONFIGS) > 0
""")

    stage_mod = _import_module("test_mod_stage_v8")

    with pytest.raises(TypeError, match="Cannot fingerprint module attribute"):
        fingerprint.get_stage_fingerprint(stage_mod.run_stage)


def test_primitive_collection_module_attr_fingerprinting(module_dir: pathlib.Path) -> None:
    """Primitive collections (dict/list/tuple/set of primitives) are supported."""
    helpers_py = module_dir / "test_mod_helpers_v9.py"
    helpers_py.write_text("""
# Primitive collections - should be fingerprinted
AGENTS = {"agent1": "config1", "agent2": "config2"}
NUMBERS = [1, 2, 3, 4, 5]
NESTED = {"key": [1, 2, {"inner": "value"}]}
""")

    stage_py = module_dir / "test_mod_stage_v9.py"
    stage_py.write_text("""
import test_mod_helpers_v9 as helpers

def run_stage():
    # Access all primitive collection attrs
    return list(helpers.AGENTS.keys()) + helpers.NUMBERS + list(helpers.NESTED.keys())
""")

    stage_mod = _import_module("test_mod_stage_v9")
    manifest = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Should have entries for the primitive collections
    assert "mod:helpers.AGENTS" in manifest
    assert "mod:helpers.NUMBERS" in manifest
    assert "mod:helpers.NESTED" in manifest

    # Values should be hashes (not repr strings)
    for key in ["mod:helpers.AGENTS", "mod:helpers.NUMBERS", "mod:helpers.NESTED"]:
        assert len(manifest[key]) == 16  # xxhash64 hex digest length


def test_primitive_collection_change_detected(module_dir: pathlib.Path) -> None:
    """Changing a primitive collection module attribute changes fingerprint."""
    helpers_py = module_dir / "test_mod_helpers_v10.py"
    helpers_py.write_text("""
AGENTS = {"agent1": "config1", "agent2": "config2"}
""")

    stage_py = module_dir / "test_mod_stage_v10.py"
    stage_py.write_text("""
import test_mod_helpers_v10 as helpers

def run_stage():
    return list(helpers.AGENTS.keys())
""")

    stage_mod = _import_module("test_mod_stage_v10")
    fp1 = fingerprint.get_stage_fingerprint(stage_mod.run_stage)
    hash1 = fp1["mod:helpers.AGENTS"]

    # Modify the collection
    helpers_py.write_text("""
AGENTS = {"agent1": "config1", "agent3": "config3"}  # CHANGED!
""")

    # Force re-import
    _reimport_module("test_mod_helpers_v10")
    stage_mod_v2 = _reimport_module("test_mod_stage_v10")
    fp2 = fingerprint.get_stage_fingerprint(stage_mod_v2.run_stage)
    hash2 = fp2["mod:helpers.AGENTS"]

    # Fingerprint MUST be different
    assert hash1 != hash2, f"Fingerprint must change when collection changes: {hash1} vs {hash2}"


def test_primitive_collection_fingerprint_deterministic(module_dir: pathlib.Path) -> None:
    """Same primitive collection produces same fingerprint (stability check)."""
    helpers_py = module_dir / "test_mod_helpers_v11.py"
    helpers_py.write_text("""
# Test determinism with various collection types
DICT_DATA = {"z": 1, "a": 2, "m": 3}  # Dict with unsorted keys
SET_DATA = {5, 1, 3, 2, 4}  # Set (unordered)
FROZENSET_DATA = frozenset([3, 1, 4, 1, 5])  # Frozenset (unordered, with duplicate)
""")

    stage_py = module_dir / "test_mod_stage_v11.py"
    stage_py.write_text("""
import test_mod_helpers_v11 as helpers

def run_stage():
    return helpers.DICT_DATA and helpers.SET_DATA and helpers.FROZENSET_DATA
""")

    stage_mod = _import_module("test_mod_stage_v11")
    fp1 = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Get fingerprint again without changes
    fp2 = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Hashes must be identical (deterministic)
    assert fp1["mod:helpers.DICT_DATA"] == fp2["mod:helpers.DICT_DATA"]
    assert fp1["mod:helpers.SET_DATA"] == fp2["mod:helpers.SET_DATA"]
    assert fp1["mod:helpers.FROZENSET_DATA"] == fp2["mod:helpers.FROZENSET_DATA"]


def test_primitive_collection_edge_cases(module_dir: pathlib.Path) -> None:
    """Empty and deeply nested primitive collections are supported."""
    helpers_py = module_dir / "test_mod_helpers_v12.py"
    helpers_py.write_text("""
# Edge cases
EMPTY_LIST = []
EMPTY_DICT = {}
EMPTY_SET = set()
EMPTY_TUPLE = ()
DEEPLY_NESTED = {"a": {"b": {"c": {"d": [1, 2, {"e": "value"}]}}}}
LARGE_LIST = list(range(200))  # 200 elements
""")

    stage_py = module_dir / "test_mod_stage_v12.py"
    stage_py.write_text("""
import test_mod_helpers_v12 as helpers

def run_stage():
    return (
        helpers.EMPTY_LIST
        or helpers.EMPTY_DICT
        or helpers.EMPTY_SET
        or helpers.EMPTY_TUPLE
        or helpers.DEEPLY_NESTED
        or helpers.LARGE_LIST
    )
""")

    stage_mod = _import_module("test_mod_stage_v12")
    manifest = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # All edge cases should be fingerprinted successfully
    assert "mod:helpers.EMPTY_LIST" in manifest
    assert "mod:helpers.EMPTY_DICT" in manifest
    assert "mod:helpers.EMPTY_SET" in manifest
    assert "mod:helpers.EMPTY_TUPLE" in manifest
    assert "mod:helpers.DEEPLY_NESTED" in manifest
    assert "mod:helpers.LARGE_LIST" in manifest

    # All should be hashes
    for key in [
        "mod:helpers.EMPTY_LIST",
        "mod:helpers.EMPTY_DICT",
        "mod:helpers.EMPTY_SET",
        "mod:helpers.EMPTY_TUPLE",
        "mod:helpers.DEEPLY_NESTED",
        "mod:helpers.LARGE_LIST",
    ]:
        assert len(manifest[key]) == 16


def test_all_primitive_types_supported(module_dir: pathlib.Path) -> None:
    """All primitive types (bool/int/float/str/bytes/None) in all collection types are supported."""
    helpers_py = module_dir / "test_mod_helpers_v13.py"
    helpers_py.write_text("""
# All primitive types in various collections
LIST_ALL_TYPES = [True, 42, 3.14, "text", b"bytes", None]
TUPLE_ALL_TYPES = (False, -1, -2.5, "tuple", b"data", None)
SET_PRIMITIVES = {1, 2, 3, "a", "b"}  # Set can't contain mutable types
FROZENSET_PRIMITIVES = frozenset([True, False, 0, 1])
DICT_ALL_TYPES = {
    "bool": True,
    "int": 123,
    "float": 45.67,
    "str": "value",
    "bytes": b"raw",
    "none": None,
}
""")

    stage_py = module_dir / "test_mod_stage_v13.py"
    stage_py.write_text("""
import test_mod_helpers_v13 as helpers

def run_stage():
    return (
        helpers.LIST_ALL_TYPES
        or helpers.TUPLE_ALL_TYPES
        or helpers.SET_PRIMITIVES
        or helpers.FROZENSET_PRIMITIVES
        or helpers.DICT_ALL_TYPES
    )
""")

    stage_mod = _import_module("test_mod_stage_v13")
    manifest = fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # All should be fingerprinted
    assert "mod:helpers.LIST_ALL_TYPES" in manifest
    assert "mod:helpers.TUPLE_ALL_TYPES" in manifest
    assert "mod:helpers.SET_PRIMITIVES" in manifest
    assert "mod:helpers.FROZENSET_PRIMITIVES" in manifest
    assert "mod:helpers.DICT_ALL_TYPES" in manifest

    # All should be hashes
    for key in [
        "mod:helpers.LIST_ALL_TYPES",
        "mod:helpers.TUPLE_ALL_TYPES",
        "mod:helpers.SET_PRIMITIVES",
        "mod:helpers.FROZENSET_PRIMITIVES",
        "mod:helpers.DICT_ALL_TYPES",
    ]:
        assert len(manifest[key]) == 16


def test_unsupported_types_comprehensive_errors(module_dir: pathlib.Path) -> None:
    """Various unsupported types in collections raise clear TypeErrors."""
    # Test 1: Mixed collection with custom object
    helpers_py = module_dir / "test_mod_helpers_v14.py"
    helpers_py.write_text("""
class Config:
    pass

MIXED = [1, 2, Config()]  # Primitive + custom object
""")

    stage_py = module_dir / "test_mod_stage_v14.py"
    stage_py.write_text("""
import test_mod_helpers_v14 as helpers

def run_stage():
    return len(helpers.MIXED) > 0
""")

    stage_mod = _import_module("test_mod_stage_v14")
    with pytest.raises(TypeError, match="Cannot fingerprint module attribute"):
        fingerprint.get_stage_fingerprint(stage_mod.run_stage)

    # Test 2: Dict containing callable (module attributes don't support callable extraction)
    helpers_py.write_text("""
def callback():
    return None

CALLBACKS = {"func": callback}
""")
    stage_py.write_text("""
import test_mod_helpers_v14 as helpers

def run_stage():
    return helpers.CALLBACKS
""")
    _reimport_module("test_mod_helpers_v14")
    stage_mod_v2 = _reimport_module("test_mod_stage_v14")

    # Module attribute collections with callables are not supported
    with pytest.raises(TypeError, match="Cannot fingerprint module attribute"):
        fingerprint.get_stage_fingerprint(stage_mod_v2.run_stage)

    # Test 3: Deeply nested unsupported type
    helpers_py.write_text("""
class DeepConfig:
    pass

NESTED_BAD = {"level1": {"level2": [1, 2, DeepConfig()]}}
""")
    stage_py.write_text("""
import test_mod_helpers_v14 as helpers

def run_stage():
    return helpers.NESTED_BAD
""")
    _reimport_module("test_mod_helpers_v14")
    stage_mod_v3 = _reimport_module("test_mod_stage_v14")

    with pytest.raises(TypeError, match="Cannot fingerprint module attribute"):
        fingerprint.get_stage_fingerprint(stage_mod_v3.run_stage)


def test_circular_reference_in_collection_raises_error(module_dir: pathlib.Path) -> None:
    """Circular references in module-level collections should raise TypeError."""
    helpers_py = module_dir / "test_mod_helpers_circular.py"
    helpers_py.write_text("""
CIRCULAR = [1, 2, 3]
CIRCULAR.append(CIRCULAR)
""")

    stage_py = module_dir / "test_mod_stage_circular.py"
    stage_py.write_text("""
import test_mod_helpers_circular as helpers

def run_stage():
    return helpers.CIRCULAR[0]
""")

    stage_mod = _import_module("test_mod_stage_circular")

    # Circular reference fails _is_primitive_collection check, so raises TypeError
    with pytest.raises(TypeError, match="Cannot fingerprint module attribute"):
        fingerprint.get_stage_fingerprint(stage_mod.run_stage)
