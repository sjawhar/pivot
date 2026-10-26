import importlib
import logging
import pathlib
import sys
import types
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, fingerprint

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
    helper_key = "func:test_mod_helpers_v1.process"
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
    assert "func:test_mod_leaf_v3.leaf_func" in fp1, (
        f"Should capture transitive dep, got: {fp1.keys()}"
    )

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
    assert fp1["func:test_mod_leaf_v3.leaf_func"] != fp2["func:test_mod_leaf_v3.leaf_func"], (
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
    assert not any("json" in key for key in fp if not key.startswith("self:")), (
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
    assert "func:test_mod_helpers_v5.helper_func" in fp1, (
        f"Should capture helper_func, got: {fp1.keys()}"
    )
    hash1 = fp1["func:test_mod_helpers_v5.helper_func"]

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
    hash2 = fp2["func:test_mod_helpers_v5.helper_func"]
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
    key = "const:test_mod_helpers_v6.THRESHOLD"
    assert key in fp, f"Should capture constant, got: {fp.keys()}"
    assert fp[key] == "0.5", f"Constant value should be repr, got: {fp[key]}"


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
    assert "func:test_mod_helpers_v7.func_a" in fp, "Should capture direct import"
    assert "func:test_mod_helpers_v7.func_b" in fp, "Should capture module attr"

    assert len(fp["func:test_mod_helpers_v7.func_a"]) == 16, "Direct import should be hashed"
    assert len(fp["func:test_mod_helpers_v7.func_b"]) == 16, "Module attr should be hashed"


@pytest.mark.parametrize(
    ("module_source", "type_name"),
    [
        pytest.param("VALUE = (1, [2])", "tuple", id="tuple-of-list"),
        pytest.param("VALUE = frozenset({(1,)})", "frozenset", id="frozenset-of-tuple"),
        pytest.param("VALUE = [1, 2]", "list", id="list"),
        pytest.param("VALUE = {1, 2}", "set", id="set"),
        pytest.param("VALUE = {'one': 1}", "dict", id="dict"),
        pytest.param(
            "import enum\n\nclass Mode(enum.Enum):\n    DEFAULT = 'default'\n\nVALUE = Mode.DEFAULT",
            "Mode",
            id="enum-member",
        ),
        pytest.param(
            "import dataclasses\n\n@dataclasses.dataclass(frozen=True)\nclass Config:\n"
            + "    value: int\n\nVALUE = Config(1)",
            "Config",
            id="frozen-dataclass",
        ),
        pytest.param(
            "import pydantic\n\nclass Config(pydantic.BaseModel):\n"
            + "    model_config = pydantic.ConfigDict(frozen=True)\n    value: int\n\nVALUE = Config(value=1)",
            "Config",
            id="frozen-pydantic",
        ),
        pytest.param(
            "class Config:\n    pass\n\nVALUE = Config()",
            "Config",
            id="class-instance",
        ),
        pytest.param(
            "class CallableConfig:\n    def __call__(self):\n        return 1\n\nVALUE = CallableConfig()",
            "CallableConfig",
            id="callable-instance",
        ),
        pytest.param(
            "class Scaler:\n    def __init__(self, factor):\n        self.factor = factor\n\n"
            + "    def apply(self, x):\n        return x * self.factor\n\nVALUE = Scaler(3).apply",
            "method",
            id="bound-method",
        ),
    ],
)
def test_non_scalar_module_attribute_raises(
    module_dir: pathlib.Path,
    module_source: str,
    type_name: str,
) -> None:
    helper_name = f"test_mod_d2_helpers_{type_name.lower()}"
    stage_name = f"test_mod_d2_stage_{type_name.lower()}"
    (module_dir / f"{helper_name}.py").write_text(f"{module_source}\n")
    (module_dir / f"{stage_name}.py").write_text(
        f"import {helper_name} as helpers\n\ndef run_stage():\n    return helpers.VALUE\n"
    )

    stage_module = _import_module(stage_name)

    with pytest.raises(exceptions.StageDefinitionError) as exc:
        fingerprint.get_stage_fingerprint_cached("train", stage_module.run_stage)

    message = str(exc.value)
    assert f"{helper_name}.VALUE" in message
    assert f"type: {type_name}" in message
    assert "Fix: move this value into the stage's StageParams" in message


def test_unsafe_fingerprinting_warns_for_non_scalar_module_attribute(
    module_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    helper_name = "test_mod_d2_unsafe_helpers"
    stage_name = "test_mod_d2_unsafe_stage"
    (module_dir / f"{helper_name}.py").write_text("VALUE = [1, 2]\n")
    (module_dir / f"{stage_name}.py").write_text(
        f"import {helper_name} as helpers\n\ndef run_stage():\n    return helpers.VALUE\n"
    )
    monkeypatch.setenv("PIVOT_UNSAFE_FINGERPRINTING", "1")

    with caplog.at_level(logging.WARNING):
        fingerprint.get_stage_fingerprint_cached("train", _import_module(stage_name).run_stage)

    assert any(f"{helper_name}.VALUE" in record.message for record in caplog.records), (
        "Unsafe fingerprinting should warn with the module attribute name"
    )


def _collision_module_source(value: int, default: int) -> str:
    return f"""\
from pivot.stage_def import StageParams

VALUE = {value}


def helper() -> int:
    return VALUE


class Config(StageParams):
    value: int = {default}


def run(config: Config = Config()) -> int:
    return helper() + config.value
"""


def test_colliding_module_definitions_keep_separate_manifest_entries(
    module_dir: pathlib.Path,
) -> None:
    first_name = "test_mod_collision_first"
    second_name = "test_mod_collision_second"
    stage_name = "test_mod_collision_stage"
    first_path = module_dir / f"{first_name}.py"
    second_path = module_dir / f"{second_name}.py"
    first_path.write_text(_collision_module_source(1, 1))
    second_path.write_text(_collision_module_source(2, 2))
    (module_dir / f"{stage_name}.py").write_text(
        f"import {first_name} as first\n"
        + f"import {second_name} as second\n\n"
        + "def run_stage() -> int:\n"
        + "    return first.run() + second.run()\n"
    )

    first_fingerprint = fingerprint.get_stage_fingerprint(_import_module(stage_name).run_stage)

    first_path.write_text(_collision_module_source(10, 10))
    _reimport_module(first_name)
    updated_stage = _reimport_module(stage_name)
    second_fingerprint = fingerprint.get_stage_fingerprint(updated_stage.run_stage)

    assert first_fingerprint != second_fingerprint
    expected_keys = {
        f"const:{first_name}.VALUE",
        f"func:{first_name}.helper",
        f"class:{first_name}.Config",
        f"schema:{first_name}.Config",
        f"const:{second_name}.VALUE",
        f"func:{second_name}.helper",
        f"class:{second_name}.Config",
        f"schema:{second_name}.Config",
    }
    assert expected_keys <= second_fingerprint.keys()
