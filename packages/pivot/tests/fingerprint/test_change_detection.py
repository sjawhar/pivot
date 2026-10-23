# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false, reportUnusedCallResult=false, reportAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

import importlib
import linecache
import pathlib
import sys
import types

import pytest

from pivot import exceptions, fingerprint

# --- Fixtures ---


@pytest.fixture
def module_dir(tmp_path: pathlib.Path):
    """Create temp directory for test modules and add to sys.path."""
    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))
    # Clean up ALL modules from our tests (multiple prefixes used)
    prefixes = (
        "test_change_",
        "test_nochange_",
        "test_limit_",
        "test_stable",
        "test_format",
        "test_multi",
        "test_dispatch_",
        "test_list_",
        "test_order_",
        "test_class_",
        "test_classchange_",
        "test_instance_",
        "test_instchange_",
        "test_nonlocal_",
        "test_params_",
        "test_nested_",
    )
    to_remove = [name for name in sys.modules if name.startswith(prefixes)]
    for name in to_remove:
        del sys.modules[name]


def _import_fresh(name: str) -> types.ModuleType:
    """Import module fresh, clearing any cached version."""
    if name in sys.modules:
        del sys.modules[name]
    linecache.clearcache()  # Clear source cache used by inspect.getsource()
    importlib.invalidate_caches()
    return importlib.import_module(name)


# =============================================================================
# SECTION 1: Changes that SHOULD cause cache miss (and DO)
# =============================================================================


def test_function_body_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing function body logic causes cache miss."""
    mod_py = module_dir / "test_change_body.py"
    mod_py.write_text("def stage():\n    return 42\n")

    mod = _import_fresh("test_change_body")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text("def stage():\n    return 43\n")
    mod = _import_fresh("test_change_body")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] != fp2["self:stage"], "Body change must cause cache miss"


def test_function_argument_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing function arguments causes cache miss."""
    mod_py = module_dir / "test_change_args.py"
    mod_py.write_text("def stage(x: int):\n    return x\n")

    mod = _import_fresh("test_change_args")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text("def stage(x: int, y: int = 0):\n    return x + y\n")
    mod = _import_fresh("test_change_args")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] != fp2["self:stage"], "Argument change must cause cache miss"


@pytest.mark.flaky(reruns=3)
def test_default_value_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing default argument value causes cache miss."""
    mod_py = module_dir / "test_change_default.py"
    mod_py.write_text("def stage(x: int = 10):\n    return x\n")

    mod = _import_fresh("test_change_default")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text("def stage(x: int = 20):\n    return x\n")
    mod = _import_fresh("test_change_default")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] != fp2["self:stage"], "Default value change must cause cache miss"


def test_helper_via_direct_import_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing helper imported via `from X import func` causes cache miss."""
    helpers_py = module_dir / "test_change_helpers_direct.py"
    helpers_py.write_text("def helper(x):\n    return x * 2\n")

    stage_py = module_dir / "test_change_stage_direct.py"
    stage_py.write_text(
        "from test_change_helpers_direct import helper\n\ndef stage():\n    return helper(10)\n"
    )

    mod = _import_fresh("test_change_stage_direct")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)
    hash1 = fp1["func:test_change_helpers_direct.helper"]

    helpers_py.write_text("def helper(x):\n    return x * 3\n")
    _import_fresh("test_change_helpers_direct")
    mod = _import_fresh("test_change_stage_direct")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    hash2 = fp2["func:test_change_helpers_direct.helper"]

    assert hash1 != hash2, "Direct import helper change must cause cache miss"


def test_helper_via_module_attr_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing helper accessed via `module.func()` causes cache miss."""
    helpers_py = module_dir / "test_change_helpers_mod.py"
    helpers_py.write_text("def helper(x):\n    return x * 2\n")

    stage_py = module_dir / "test_change_stage_mod.py"
    stage_py.write_text(
        "import test_change_helpers_mod as helpers\n\ndef stage():\n    return helpers.helper(10)\n"
    )

    mod = _import_fresh("test_change_stage_mod")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)
    hash1 = fp1["func:test_change_helpers_mod.helper"]

    helpers_py.write_text("def helper(x):\n    return x * 3\n")
    _import_fresh("test_change_helpers_mod")
    mod = _import_fresh("test_change_stage_mod")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    hash2 = fp2["func:test_change_helpers_mod.helper"]

    assert hash1 != hash2, "Module attr helper change must cause cache miss"


def test_transitive_dependency_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing transitive dependency (helper's helper) causes cache miss."""
    leaf_py = module_dir / "test_change_leaf.py"
    leaf_py.write_text("def leaf(x):\n    return x + 1\n")

    middle_py = module_dir / "test_change_middle.py"
    middle_py.write_text(
        "import test_change_leaf as leaf_mod\n\ndef middle(x):\n    return leaf_mod.leaf(x) * 2\n"
    )

    stage_py = module_dir / "test_change_stage_trans.py"
    stage_py.write_text(
        "import test_change_middle as mid\n\ndef stage():\n    return mid.middle(10)\n"
    )

    mod = _import_fresh("test_change_stage_trans")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)
    hash1 = fp1["func:test_change_leaf.leaf"]

    leaf_py.write_text("def leaf(x):\n    return x + 100\n")
    for m in ["test_change_leaf", "test_change_middle", "test_change_stage_trans"]:
        _import_fresh(m)
    mod = _import_fresh("test_change_stage_trans")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    hash2 = fp2["func:test_change_leaf.leaf"]

    assert hash1 != hash2, "Transitive dependency change must cause cache miss"


def test_module_constant_captured_via_module_attr(module_dir: pathlib.Path) -> None:
    """Module constants accessed via module.CONST are captured in fingerprint."""
    config_py = module_dir / "test_change_config.py"
    config_py.write_text("THRESHOLD = 0.5\n")

    stage_py = module_dir / "test_change_stage_const.py"
    stage_py.write_text(
        "import test_change_config as config\n\ndef stage(x):\n    return x > config.THRESHOLD\n"
    )

    mod = _import_fresh("test_change_stage_const")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    key = "const:test_change_config.THRESHOLD"
    assert key in fp, "Module constant should be captured"
    assert fp[key] == "0.5", "Constant value should be repr"


def test_global_constant_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing global constant in same module causes cache miss."""
    mod_py = module_dir / "test_change_global.py"
    mod_py.write_text("MULTIPLIER = 2\n\ndef stage(x):\n    return x * MULTIPLIER\n")

    mod = _import_fresh("test_change_global")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)
    val1 = fp1["const:test_change_global.MULTIPLIER"]

    mod_py.write_text("MULTIPLIER = 10\n\ndef stage(x):\n    return x * MULTIPLIER\n")
    mod = _import_fresh("test_change_global")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    val2 = fp2["const:test_change_global.MULTIPLIER"]

    assert val1 == "2"
    assert val2 == "10"
    assert val1 != val2, "Global constant change must cause cache miss"


def test_variable_rename_causes_miss(module_dir: pathlib.Path) -> None:
    """Renaming local variables causes cache miss (AST includes names)."""
    mod_py = module_dir / "test_change_varname.py"
    mod_py.write_text("def stage(x):\n    result = x * 2\n    return result\n")

    mod = _import_fresh("test_change_varname")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text("def stage(x):\n    output = x * 2\n    return output\n")
    mod = _import_fresh("test_change_varname")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] != fp2["self:stage"], "Variable rename must cause cache miss"


def test_async_sync_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Converting sync to async function causes cache miss (different AST node type)."""
    mod_py = module_dir / "test_change_async.py"
    mod_py.write_text("def stage():\n    return 42\n")

    mod = _import_fresh("test_change_async")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text("async def stage():\n    return 42\n")
    mod = _import_fresh("test_change_async")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] != fp2["self:stage"], "Async/sync change must cause cache miss"


def test_type_annotation_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Adding or changing type annotations causes cache miss (AST includes annotations)."""
    mod_py = module_dir / "test_change_types.py"
    mod_py.write_text("def stage(x):\n    return x * 2\n")

    mod = _import_fresh("test_change_types")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text("def stage(x: int) -> int:\n    return x * 2\n")
    mod = _import_fresh("test_change_types")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] != fp2["self:stage"], "Type annotation change must cause cache miss"


# =============================================================================
# SECTION 2: Changes that should NOT cause cache miss (by design)
# =============================================================================


def test_docstring_change_no_miss(module_dir: pathlib.Path) -> None:
    """Changing docstring does NOT cause cache miss (docstrings stripped)."""
    mod_py = module_dir / "test_nochange_doc.py"
    mod_py.write_text('def stage():\n    """Original docstring."""\n    return 42\n')

    mod = _import_fresh("test_nochange_doc")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text('def stage():\n    """CHANGED docstring!"""\n    return 42\n')
    mod = _import_fresh("test_nochange_doc")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] == fp2["self:stage"], "Docstring change must not cause cache miss"


def test_comment_change_no_miss(module_dir: pathlib.Path) -> None:
    """Changing comments does NOT cause cache miss (comments not in AST)."""
    mod_py = module_dir / "test_nochange_comment.py"
    mod_py.write_text("def stage():\n    # Original comment\n    return 42\n")

    mod = _import_fresh("test_nochange_comment")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text("def stage():\n    # CHANGED comment!\n    return 42\n")
    mod = _import_fresh("test_nochange_comment")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] == fp2["self:stage"], "Comment change must not cause cache miss"


def test_whitespace_change_no_miss(module_dir: pathlib.Path) -> None:
    """Changing whitespace does NOT cause cache miss (AST ignores whitespace)."""
    mod_py = module_dir / "test_nochange_ws.py"
    mod_py.write_text("def stage():\n    return 42\n")

    mod = _import_fresh("test_nochange_ws")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text("def stage():\n\n\n    return 42\n\n")
    mod = _import_fresh("test_nochange_ws")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1["self:stage"] == fp2["self:stage"], "Whitespace change must not cause cache miss"


def test_function_rename_no_miss(module_dir: pathlib.Path) -> None:
    """Renaming function does NOT cause cache miss (names normalized in AST)."""
    mod_py = module_dir / "test_nochange_rename.py"
    mod_py.write_text("def original_name():\n    return 42\n")

    mod = _import_fresh("test_nochange_rename")
    fp1 = fingerprint.get_stage_fingerprint(mod.original_name)

    mod_py.write_text("def renamed_function():\n    return 42\n")
    mod = _import_fresh("test_nochange_rename")
    fp2 = fingerprint.get_stage_fingerprint(mod.renamed_function)

    assert fp1["self:original_name"] == fp2["self:renamed_function"], (
        "Function rename must not cause cache miss (AST normalizes names)"
    )


def test_stdlib_module_attrs_not_tracked(module_dir: pathlib.Path) -> None:
    """Stdlib module attributes are NOT tracked in fingerprint."""
    mod_py = module_dir / "test_nochange_stdlib.py"
    mod_py.write_text(
        "import json as json_mod\n\ndef stage():\n    return json_mod.dumps({'key': 'value'})\n"
    )

    mod = _import_fresh("test_nochange_stdlib")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    # json is stdlib - should NOT be in fingerprint
    assert not any("json" in key for key in fp if not key.startswith("self:")), fp


# =============================================================================
# SECTION 3: Future improvements (xfail tests documenting desired behavior)
# =============================================================================
# These tests document behavior we'd LIKE to have but don't yet implement.
# They use @pytest.mark.xfail to serve as a to-do list for contributors.
# When a limitation is fixed, the test will start passing (xpass) automatically.


def test_underscore_helper_change_detected(module_dir: pathlib.Path) -> None:
    """Helpers starting with _ are tracked for change detection."""
    helpers_py = module_dir / "test_limit_helpers_us.py"
    helpers_py.write_text("def _private_helper(x):\n    return x * 2\n")

    stage_py = module_dir / "test_limit_stage_us.py"
    stage_py.write_text(
        "from test_limit_helpers_us import _private_helper\n\n"
        + "def stage():\n    return _private_helper(10)\n"
    )

    mod = _import_fresh("test_limit_stage_us")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "func:test_limit_helpers_us._private_helper" in fp, (
        "Should capture underscore-prefixed helpers"
    )


def test_loaded_body_import_is_tracked_as_a_helper(module_dir: pathlib.Path) -> None:
    """A body import already loaded during discovery is followed to its helper function."""
    helpers_py = module_dir / "test_limit_helpers_lazy.py"
    helpers_py.write_text("def helper(x):\n    return x * 2\n")

    stage_py = module_dir / "test_limit_stage_lazy.py"
    stage_py.write_text(
        "def stage():\n    from test_limit_helpers_lazy import helper\n    return helper(10)\n"
    )

    _import_fresh("test_limit_helpers_lazy")
    mod = _import_fresh("test_limit_stage_lazy")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "func:test_limit_helpers_lazy.helper" in fp, (
        "Should capture a body import that discovery already loaded"
    )


def test_unloaded_body_import_is_refused_without_running_it(module_dir: pathlib.Path) -> None:
    """A body import of a user module discovery did not load is refused, and fingerprinting
    neither imports it nor its package."""
    package = module_dir / "test_change_static_import"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helper.py").write_text("def value():\n    return 1\n")
    (package / "stages.py").write_text(
        """\
def stage():
    if False:
        from . import helper
        return helper.value()
    return 1
"""
    )

    module = _import_fresh("test_change_static_import.stages")
    with pytest.raises(
        exceptions.StageDefinitionError, match="imports user module 'test_change_static_import"
    ):
        fingerprint.get_stage_fingerprint(module.stage)
    assert "test_change_static_import.helper" not in sys.modules


# =============================================================================
# SECTION 4: Hash stability and consistency
# =============================================================================


def test_same_code_same_fingerprint(module_dir: pathlib.Path) -> None:
    """Same code produces identical fingerprint (deterministic)."""
    mod_py = module_dir / "test_stable.py"
    mod_py.write_text("def stage():\n    return 42\n")

    mod = _import_fresh("test_stable")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1 == fp2, "Same code must produce identical fingerprint"


def test_fingerprint_is_16_char_hex(module_dir: pathlib.Path) -> None:
    """Fingerprint hashes are 16-character hexadecimal strings."""
    mod_py = module_dir / "test_format.py"
    mod_py.write_text("def stage():\n    return 42\n")

    mod = _import_fresh("test_format")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    hash_val = fp["self:stage"]
    assert len(hash_val) == 16, f"Hash should be 16 chars, got {len(hash_val)}"
    assert all(c in "0123456789abcdef" for c in hash_val), f"Hash should be hex, got {hash_val}"


def test_multiple_runs_stable(module_dir: pathlib.Path) -> None:
    """Multiple fingerprint runs produce stable results."""
    mod_py = module_dir / "test_multi.py"
    mod_py.write_text(
        "import json as json_mod\n\n"
        + "CONSTANT = 42\n\n"
        + "def helper(x):\n    return x * 2\n\n"
        + "def stage():\n"
        + "    return helper(CONSTANT) + json_mod.dumps({})\n"
    )

    mod = _import_fresh("test_multi")
    fingerprints = [fingerprint.get_stage_fingerprint(mod.stage) for _ in range(5)]

    for fp in fingerprints[1:]:
        assert fp == fingerprints[0], "Fingerprints must be stable across runs"


# =============================================================================
# SECTION 5: Class definition tracking
# =============================================================================


def test_class_definition_tracked_with_class_prefix(module_dir: pathlib.Path) -> None:
    """Class definitions are tracked with 'class:' prefix."""
    helpers_py = module_dir / "test_class_helpers.py"
    helpers_py.write_text("class MyProcessor:\n    def process(self, x):\n        return x * 2\n")

    stage_py = module_dir / "test_class_stage.py"
    stage_py.write_text(
        "from test_class_helpers import MyProcessor\n\n"
        + "def stage():\n    return MyProcessor().process(10)\n"
    )

    mod = _import_fresh("test_class_stage")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "class:test_class_helpers.MyProcessor" in fp, "Should capture class with 'class:' prefix"
    assert "func:test_class_helpers.MyProcessor" not in fp, (
        "Should NOT use 'func:' prefix for classes"
    )


def test_class_definition_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing class definition causes cache miss."""
    helpers_py = module_dir / "test_classchange_helpers.py"
    helpers_py.write_text("class MyProcessor:\n    def process(self, x):\n        return x * 2\n")

    stage_py = module_dir / "test_classchange_stage.py"
    stage_py.write_text(
        "from test_classchange_helpers import MyProcessor\n\n"
        + "def stage():\n    return MyProcessor().process(10)\n"
    )

    mod = _import_fresh("test_classchange_stage")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    # Change the class method
    helpers_py.write_text(
        "class MyProcessor:\n    def process(self, x):\n        return x * 999\n"  # Changed!
    )
    _import_fresh("test_classchange_helpers")
    mod = _import_fresh("test_classchange_stage")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1 != fp2, "Class method change must cause cache miss"


# =============================================================================
# SECTION 6: StageParams tracking
# =============================================================================


def test_stageparams_property_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing @property method on StageParams subclass causes cache miss."""
    params_py = module_dir / "test_params_prop.py"
    params_py.write_text("""from pivot.stage_def import StageParams

class MyParams(StageParams):
    base_rate: float = 0.01
    multiplier: float = 2.0

    @property
    def effective_rate(self) -> float:
        return self.base_rate * self.multiplier
""")

    stage_py = module_dir / "test_params_stage.py"
    stage_py.write_text("""from test_params_prop import MyParams

def stage(params: MyParams) -> int:
    return int(params.effective_rate * 100)
""")

    mod = _import_fresh("test_params_stage")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    # Change the property implementation (minimal modification)
    original = params_py.read_text()
    params_py.write_text(original.replace("self.multiplier", "self.multiplier * 1.1"))
    _import_fresh("test_params_prop")
    mod = _import_fresh("test_params_stage")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    key = "class:test_params_prop.MyParams"
    assert fp1[key] != fp2[key], "@property change on StageParams must cause cache miss"


def test_stageparams_method_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing regular method on StageParams subclass causes cache miss."""
    params_py = module_dir / "test_params_method.py"
    params_py.write_text("""from pivot.stage_def import StageParams

class MyParams(StageParams):
    base_rate: float = 0.01

    def compute_rate(self, multiplier: float) -> float:
        return self.base_rate * multiplier
""")

    stage_py = module_dir / "test_params_method_stage.py"
    stage_py.write_text("""from test_params_method import MyParams

def stage(params: MyParams) -> int:
    return int(params.compute_rate(2.0) * 100)
""")

    mod = _import_fresh("test_params_method_stage")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    # Change the method implementation
    original = params_py.read_text()
    params_py.write_text(original.replace("* multiplier", "* multiplier * 1.1"))
    _import_fresh("test_params_method")
    mod = _import_fresh("test_params_method_stage")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    key = "class:test_params_method.MyParams"
    assert fp1[key] != fp2[key], "Method change on StageParams must cause cache miss"


def test_stageparams_class_variable_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing class variable on StageParams subclass causes cache miss."""
    params_py = module_dir / "test_params_classvar.py"
    params_py.write_text("""from typing import ClassVar
from pivot.stage_def import StageParams

class MyParams(StageParams):
    VERSION: ClassVar[str] = "1.0"
    base_rate: float = 0.01
""")

    stage_py = module_dir / "test_params_classvar_stage.py"
    stage_py.write_text("""from test_params_classvar import MyParams

def stage(params: MyParams) -> str:
    return f"{MyParams.VERSION}: {params.base_rate}"
""")

    mod = _import_fresh("test_params_classvar_stage")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    # Change the class variable
    original = params_py.read_text()
    params_py.write_text(original.replace('"1.0"', '"2.0"'))
    _import_fresh("test_params_classvar")
    mod = _import_fresh("test_params_classvar_stage")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    key = "class:test_params_classvar.MyParams"
    assert fp1[key] != fp2[key], "Class variable change on StageParams must cause cache miss"


# =============================================================================
# SECTION 7: Nested function global references
# =============================================================================


def test_nested_function_global_reference_detected(module_dir: pathlib.Path) -> None:
    """Nested function globals are captured in fingerprint manifest."""
    mod_py = module_dir / "test_nested_global_ref.py"
    mod_py.write_text(
        """def helper(x):
    return x + 1

def compute():
    def process_batch(items):
        return [helper(i) for i in items]
    return process_batch([1, 2, 3])

def stage():
    return compute()
"""
    )

    mod = _import_fresh("test_nested_global_ref")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "func:test_nested_global_ref.helper" in fp, "Nested global helper should be captured"


def test_nested_function_global_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing nested-only global helper causes fingerprint change."""
    mod_py = module_dir / "test_nested_global_change.py"
    mod_py.write_text(
        """def helper(x):
    return x + 1

def compute():
    def process_batch(items):
        return [helper(i) for i in items]
    return process_batch([1, 2, 3])

def stage():
    return compute()
"""
    )

    mod = _import_fresh("test_nested_global_change")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    mod_py.write_text(
        """def helper(x):
    return x + 999

def compute():
    def process_batch(items):
        return [helper(i) for i in items]
    return process_batch([1, 2, 3])

def stage():
    return compute()
"""
    )
    mod = _import_fresh("test_nested_global_change")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1 != fp2, "Nested global helper change must cause cache miss"


def test_deeply_nested_function_global_detected(module_dir: pathlib.Path) -> None:
    """Deeply nested globals are captured in fingerprint manifest."""
    mod_py = module_dir / "test_nested_deep.py"
    mod_py.write_text(
        """def leaf(x):
    return x + 1

def tweak(x):
    return x * 2

def outer():
    def middle():
        def inner():
            return leaf(1)
        return tweak(inner())
    return middle()

def stage():
    return outer()
"""
    )

    mod = _import_fresh("test_nested_deep")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "func:test_nested_deep.leaf" in fp, "Deeply nested leaf should be captured"
    assert "func:test_nested_deep.tweak" in fp, "Deeply nested tweak should be captured"


def test_nested_lambda_global_detected(module_dir: pathlib.Path) -> None:
    """Nested lambda globals are captured in fingerprint manifest."""
    mod_py = module_dir / "test_nested_lambda.py"
    mod_py.write_text(
        """def processor(x):
    return x * 2

def compute(items):
    fn = lambda x: processor(x)
    return [fn(item) for item in items]

def stage():
    return compute([1, 2])
"""
    )

    mod = _import_fresh("test_nested_lambda")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "func:test_nested_lambda.processor" in fp, "Nested lambda global should be captured"


def test_nested_function_builtin_not_tracked(module_dir: pathlib.Path) -> None:
    """Builtins used only in nested functions are not tracked."""
    mod_py = module_dir / "test_nested_builtin.py"
    mod_py.write_text(
        """def compute():
    def inner(items):
        return min(items) + len(items) + sum(range(3))
    return inner([1, 2, 3])

def stage():
    return compute()
"""
    )

    mod = _import_fresh("test_nested_builtin")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "func:min" not in fp, "Builtins should not be tracked as globals"
    assert "func:len" not in fp, "Builtins should not be tracked as globals"


def test_nested_function_attribute_collision_over_invalidates(
    module_dir: pathlib.Path,
) -> None:
    """Attribute name matching a global is over-included (known benign behavior).

    When a nested function does `text.upper()`, 'upper' appears in co_names.
    If a module-level function also named 'upper' exists in __globals__,
    it gets included in the fingerprint even though the nested function
    only uses it as an attribute.  This is a known benign over-inclusion:
    co_names is a superset of LOAD_GLOBAL names, and over-invalidation
    is safe (under-invalidation is catastrophic).
    """
    mod_py = module_dir / "test_nested_attr_collision.py"
    mod_py.write_text(
        """def upper(x):
    \"\"\"Module-level function that shares name with str.upper.\"\"\"
    return x * 2

def compute(text):
    def nested():
        # text.upper() puts 'upper' in co_names as attribute access,
        # but module-level 'upper' also exists in __globals__
        return text.upper()
    return nested()

def stage():
    return compute("hello")
"""
    )

    mod = _import_fresh("test_nested_attr_collision")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    # 'upper' is over-included because co_names contains attribute access names
    # and __globals__ has a function with the same name. This is benign:
    # over-invalidation is safe, under-invalidation is catastrophic.
    assert "func:test_nested_attr_collision.upper" in fp, (
        "Attribute name colliding with global should be over-included (benign)"
    )


def test_method_global_change_causes_miss(module_dir: pathlib.Path) -> None:
    """A module global read inside a user class's method is part of the stage's identity."""
    mod_py = module_dir / "test_change_method_global.py"
    body = (
        "class Scaler:\n"
        "    def apply(self, v):\n"
        "        return v * SCALE\n"
        "def stage():\n"
        "    return Scaler().apply(1)\n"
    )
    mod_py.write_text("SCALE = 2\n" + body)
    fp1 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_method_global").stage)
    # Different size, so Python's mtime+size bytecode check cannot reuse stale bytecode.
    mod_py.write_text("SCALE = 30\n" + body)
    fp2 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_method_global").stage)
    assert fp1 != fp2, "editing a global a method reads must change the stage fingerprint"


def test_method_regex_change_causes_miss(module_dir: pathlib.Path) -> None:
    """A module-level compiled regex read only inside a method of a user class, which the stage
    calls through an instance of that class imported from another module, is part of the
    stage's identity."""
    lib_py = module_dir / "test_change_method_regex_lib.py"
    body = (
        "class ScopeWalk:\n"
        "    def step(self, code):\n"
        "        return LOCALITY.match(code) is not None\n"
    )
    lib_py.write_text("import re\nLOCALITY = re.compile(r'^Local\\s+Set')\n" + body)
    (module_dir / "test_change_method_regex.py").write_text(
        "from test_change_method_regex_lib import ScopeWalk\n"
        + "def stage():\n"
        + "    return ScopeWalk().step('Local Set')\n"
    )
    fp1 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_method_regex").stage)
    lib_py.write_text("import re\nLOCALITY = re.compile(r'^NEVER')\n" + body)
    _import_fresh("test_change_method_regex_lib")
    fp2 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_method_regex").stage)
    assert fp1 != fp2, "editing the regex a method reads must change the stage fingerprint"


def test_method_helper_change_causes_miss(module_dir: pathlib.Path) -> None:
    """A helper called only from a user class's method is part of the stage's identity."""
    helper_py = module_dir / "test_change_method_helper_lib.py"
    helper_py.write_text("def bump(v):\n    return v + 1\n")
    (module_dir / "test_change_method_helper.py").write_text(
        "from test_change_method_helper_lib import bump\n"
        + "class Runner:\n"
        + "    @classmethod\n"
        + "    def go(cls):\n"
        + "        return bump(1)\n"
        + "def stage():\n"
        + "    return Runner.go()\n"
    )
    fp1 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_method_helper").stage)
    helper_py.write_text("def bump(v):\n    return v + 100\n")
    _import_fresh("test_change_method_helper_lib")
    fp2 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_method_helper").stage)
    assert fp1 != fp2, "editing a helper a method calls must change the stage fingerprint"


def test_in_body_import_change_causes_miss(module_dir: pathlib.Path) -> None:
    """A module discovery loaded and the stage imports in its body is a dependency."""
    helper_py = module_dir / "test_change_inbody_lib.py"
    helper_py.write_text("def value():\n    return 2\n")
    (module_dir / "test_change_inbody.py").write_text(
        "def stage():\n    import test_change_inbody_lib as lib\n    return lib.value()\n"
    )
    _import_fresh("test_change_inbody_lib")
    fp1 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_inbody").stage)
    helper_py.write_text("def value():\n    return 101\n")
    _import_fresh("test_change_inbody_lib")
    fp2 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_inbody").stage)
    assert fp1 != fp2, "editing a module the stage imports in its body must change it"


def test_dataclass_method_reading_global_is_walked(module_dir: pathlib.Path) -> None:
    """A dataclass in the stage's annotations has its methods walked like any user class's, so
    a global one of them reads is tracked, not refused as a method on a data class."""
    mod_py = module_dir / "test_change_dataclass_method.py"
    body = (
        "import dataclasses\n"
        "@dataclasses.dataclass\n"
        "class Settings:\n"
        "    width: int = 1\n"
        "    def slots(self):\n"
        "        return self.width * FACTOR\n"
        "def stage(settings: Settings) -> int:\n"
        "    return settings.slots()\n"
    )
    mod_py.write_text("FACTOR = 2\n" + body)
    fp1 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_dataclass_method").stage)
    mod_py.write_text("FACTOR = 20\n" + body)
    fp2 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_dataclass_method").stage)
    assert fp1 != fp2, "editing a global a dataclass method reads must change the fingerprint"


def test_in_body_from_import_of_non_scalar_is_refused(module_dir: pathlib.Path) -> None:
    """`from X import NAME` in a stage body, where NAME is a mutable module value, is refused
    as the same value read as a global is."""
    (module_dir / "test_change_inbody_list_lib.py").write_text("SUFFIXES = ['!']\n")
    (module_dir / "test_change_inbody_list.py").write_text(
        "def stage():\n"
        + "    from test_change_inbody_list_lib import SUFFIXES\n"
        + "    return SUFFIXES[0]\n"
    )
    _import_fresh("test_change_inbody_list_lib")
    module = _import_fresh("test_change_inbody_list")
    with pytest.raises(exceptions.StageDefinitionError, match="SUFFIXES"):
        fingerprint.get_stage_fingerprint(module.stage)


def test_relative_in_body_import_change_causes_miss(module_dir: pathlib.Path) -> None:
    """`from .sibling import value` in a stage body is resolved against the stage's package."""
    package = module_dir / "test_change_relpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    sibling_py = package / "sibling.py"
    sibling_py.write_text("def value():\n    return 2\n")
    (package / "stages.py").write_text(
        "def stage():\n    from .sibling import value\n    return value()\n"
    )
    _import_fresh("test_change_relpkg.sibling")
    fp1 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_relpkg.stages").stage)
    sibling_py.write_text("def value():\n    return 101\n")
    _import_fresh("test_change_relpkg.sibling")
    fp2 = fingerprint.get_stage_fingerprint(_import_fresh("test_change_relpkg.stages").stage)
    assert fp1 != fp2, "editing a module a relative in-body import names must change it"
