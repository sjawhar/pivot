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
    """Unsupported types (like lists, arrays) in module attrs raise TypeError."""
    helpers_py = module_dir / "test_mod_helpers_v8.py"
    helpers_py.write_text("""
# This is an unsupported type for module attribute fingerprinting
MY_LIST = [1, 2, 3, 4, 5]

def process(x):
    return x in MY_LIST
""")

    stage_py = module_dir / "test_mod_stage_v8.py"
    stage_py.write_text("""
import test_mod_helpers_v8 as helpers

def run_stage():
    # Uses the list via module attribute access
    return helpers.process(3) and len(helpers.MY_LIST) > 0
""")

    stage_mod = _import_module("test_mod_stage_v8")

    with pytest.raises(TypeError, match="Cannot fingerprint module attribute"):
        fingerprint.get_stage_fingerprint(stage_mod.run_stage)
