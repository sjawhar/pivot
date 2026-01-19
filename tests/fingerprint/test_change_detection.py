import importlib
import linecache
import pathlib
import sys
import types

import pytest

from pivot import fingerprint

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
    hash1 = fp1["func:helper"]

    helpers_py.write_text("def helper(x):\n    return x * 3\n")
    _import_fresh("test_change_helpers_direct")
    mod = _import_fresh("test_change_stage_direct")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    hash2 = fp2["func:helper"]

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
    hash1 = fp1["mod:helpers.helper"]

    helpers_py.write_text("def helper(x):\n    return x * 3\n")
    _import_fresh("test_change_helpers_mod")
    mod = _import_fresh("test_change_stage_mod")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    hash2 = fp2["mod:helpers.helper"]

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
    hash1 = fp1["mod:leaf_mod.leaf"]

    leaf_py.write_text("def leaf(x):\n    return x + 100\n")
    for m in ["test_change_leaf", "test_change_middle", "test_change_stage_trans"]:
        _import_fresh(m)
    mod = _import_fresh("test_change_stage_trans")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    hash2 = fp2["mod:leaf_mod.leaf"]

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

    assert "mod:config.THRESHOLD" in fp, "Module constant should be captured"
    assert fp["mod:config.THRESHOLD"] == "0.5", "Constant value should be repr"


def test_global_constant_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing global constant in same module causes cache miss."""
    mod_py = module_dir / "test_change_global.py"
    mod_py.write_text("MULTIPLIER = 2\n\ndef stage(x):\n    return x * MULTIPLIER\n")

    mod = _import_fresh("test_change_global")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)
    val1 = fp1["const:MULTIPLIER"]

    mod_py.write_text("MULTIPLIER = 10\n\ndef stage(x):\n    return x * MULTIPLIER\n")
    mod = _import_fresh("test_change_global")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    val2 = fp2["const:MULTIPLIER"]

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


def test_stdlib_function_marked_callable(module_dir: pathlib.Path) -> None:
    """Stdlib functions are marked 'callable', not hashed (version changes ignored)."""
    mod_py = module_dir / "test_nochange_stdlib.py"
    mod_py.write_text(
        "import json as json_mod\n\ndef stage():\n    return json_mod.dumps({'key': 'value'})\n"
    )

    mod = _import_fresh("test_nochange_stdlib")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp["mod:json_mod.dumps"] == "callable", "Stdlib functions must be 'callable', not hashed"


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

    assert "func:_private_helper" in fp, "Should capture underscore-prefixed helpers"


@pytest.mark.xfail(reason="Lazy imports inside function body not analyzed by getclosurevars()")
def test_lazy_import_change_detected(module_dir: pathlib.Path) -> None:
    """Imports inside function body should be tracked for change detection."""
    helpers_py = module_dir / "test_limit_helpers_lazy.py"
    helpers_py.write_text("def helper(x):\n    return x * 2\n")

    stage_py = module_dir / "test_limit_stage_lazy.py"
    stage_py.write_text(
        "def stage():\n    from test_limit_helpers_lazy import helper\n    return helper(10)\n"
    )

    mod = _import_fresh("test_limit_stage_lazy")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "func:helper" in fp, "Should capture lazy imports inside function body"


@pytest.mark.xfail(
    reason="Class instances not modules; class definitions not tracked even when imported"
)
def test_class_instance_method_change_detected(module_dir: pathlib.Path) -> None:
    """Class instance method changes should be tracked for change detection."""
    helpers_py = module_dir / "test_limit_helpers_method.py"
    helpers_py.write_text("class Processor:\n    def process(self, x):\n        return x * 2\n")

    stage_py = module_dir / "test_limit_stage_method.py"
    stage_py.write_text(
        "from test_limit_helpers_method import Processor\n\n"
        + "processor = Processor()\n\n"
        + "def stage():\n    return processor.process(10)\n"
    )

    mod = _import_fresh("test_limit_stage_method")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "mod:processor.process" in fp, "Should capture instance method usage"


def test_dynamic_dispatch_change_detected(module_dir: pathlib.Path) -> None:
    """Functions in dispatch dicts are tracked for change detection."""
    helpers_py = module_dir / "test_limit_helpers_dyn.py"
    helpers_py.write_text(
        "def add(x):\n    return x + 1\n\n"
        + "def mul(x):\n    return x * 2\n\n"
        + "FUNCS = {'add': add, 'mul': mul}\n"
    )

    stage_py = module_dir / "test_limit_stage_dyn.py"
    stage_py.write_text(
        "from test_limit_helpers_dyn import FUNCS\n\n"
        + "def stage(op='add'):\n    return FUNCS[op](10)\n"
    )

    mod = _import_fresh("test_limit_stage_dyn")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    # Collection callables are tracked with key format: func:COLLECTION_NAME['key']
    assert "func:FUNCS['add']" in fp, "Should capture functions in dispatch dicts"
    assert "func:FUNCS['mul']" in fp, "Should capture functions in dispatch dicts"


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


def test_dispatch_dict_function_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing function inside dispatch dict causes cache miss."""
    helpers_py = module_dir / "test_dispatch_helpers.py"
    helpers_py.write_text(
        "def add(x):\n    return x + 1\n\n"
        + "def mul(x):\n    return x * 2\n\n"
        + "FUNCS = {'add': add, 'mul': mul}\n"
    )

    stage_py = module_dir / "test_dispatch_stage.py"
    stage_py.write_text(
        "from test_dispatch_helpers import FUNCS\n\n"
        + "def stage(op='add'):\n    return FUNCS[op](10)\n"
    )

    mod = _import_fresh("test_dispatch_stage")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)
    hash1 = fp1["func:FUNCS['add']"]

    # Change the add function
    helpers_py.write_text(
        "def add(x):\n    return x + 999\n\n"  # Changed!
        + "def mul(x):\n    return x * 2\n\n"
        + "FUNCS = {'add': add, 'mul': mul}\n"
    )
    _import_fresh("test_dispatch_helpers")
    mod = _import_fresh("test_dispatch_stage")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)
    hash2 = fp2["func:FUNCS['add']"]

    assert hash1 != hash2, "Dispatch dict function change must cause cache miss"


def test_list_callable_tracking(module_dir: pathlib.Path) -> None:
    """Functions in lists are tracked for change detection."""
    helpers_py = module_dir / "test_list_helpers.py"
    helpers_py.write_text(
        "def step1(x):\n    return x + 1\n\n"
        + "def step2(x):\n    return x * 2\n\n"
        + "PIPELINE = [step1, step2]\n"
    )

    stage_py = module_dir / "test_list_stage.py"
    stage_py.write_text(
        "from test_list_helpers import PIPELINE\n\n"
        + "def stage(x):\n"
        + "    for step in PIPELINE:\n"
        + "        x = step(x)\n"
        + "    return x\n"
    )

    mod = _import_fresh("test_list_stage")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "func:PIPELINE[0]" in fp, "Should capture first function in list"
    assert "func:PIPELINE[1]" in fp, "Should capture second function in list"


def test_fingerprint_ordering_stability(module_dir: pathlib.Path) -> None:
    """Fingerprint is stable regardless of dict key ordering in source."""
    # Test that different dict key ORDER in source produces same fingerprint
    # (because we sort keys during processing)
    helpers_py = module_dir / "test_order_helpers.py"

    # Version 1: dict literal has 'add' first
    helpers_py.write_text(
        "def add(x):\n    return x + 1\n\n"
        + "def mul(x):\n    return x * 2\n\n"
        + "FUNCS = {'add': add, 'mul': mul}\n"
    )
    stage_py = module_dir / "test_order_stage.py"
    stage_py.write_text(
        "from test_order_helpers import FUNCS\n\n" + "def stage():\n    return FUNCS\n"
    )

    mod = _import_fresh("test_order_stage")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    # Version 2: dict literal has 'mul' first (same functions, different key order)
    helpers_py.write_text(
        "def add(x):\n    return x + 1\n\n"
        + "def mul(x):\n    return x * 2\n\n"
        + "FUNCS = {'mul': mul, 'add': add}\n"  # Only dict key order changed
    )
    _import_fresh("test_order_helpers")
    mod = _import_fresh("test_order_stage")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    # Same functions, same behavior - fingerprints should match
    assert fp1 == fp2, "Fingerprint should be stable regardless of dict key order"


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

    assert "class:MyProcessor" in fp, "Should capture class with 'class:' prefix"
    assert "func:MyProcessor" not in fp, "Should NOT use 'func:' prefix for classes"


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


def test_class_instance_tracked(module_dir: pathlib.Path) -> None:
    """Module-level class instances have their class tracked."""
    helpers_py = module_dir / "test_instance_helpers.py"
    helpers_py.write_text(
        "class Processor:\n"
        + "    def process(self, x):\n"
        + "        return x * 2\n\n"
        + "processor = Processor()\n"
    )

    stage_py = module_dir / "test_instance_stage.py"
    stage_py.write_text(
        "from test_instance_helpers import processor\n\n"
        + "def stage():\n    return processor.process(10)\n"
    )

    mod = _import_fresh("test_instance_stage")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "class:processor.__class__" in fp, "Should capture instance's class"


def test_class_instance_change_causes_miss(module_dir: pathlib.Path) -> None:
    """Changing class of a module-level instance causes cache miss."""
    helpers_py = module_dir / "test_instchange_helpers.py"
    helpers_py.write_text(
        "class Processor:\n"
        + "    def process(self, x):\n"
        + "        return x * 2\n\n"
        + "processor = Processor()\n"
    )

    stage_py = module_dir / "test_instchange_stage.py"
    stage_py.write_text(
        "from test_instchange_helpers import processor\n\n"
        + "def stage():\n    return processor.process(10)\n"
    )

    mod = _import_fresh("test_instchange_stage")
    fp1 = fingerprint.get_stage_fingerprint(mod.stage)

    # Change the class method
    helpers_py.write_text(
        "class Processor:\n"
        + "    def process(self, x):\n"
        + "        return x * 999\n\n"
        + "processor = Processor()\n"
    )
    _import_fresh("test_instchange_helpers")
    mod = _import_fresh("test_instchange_stage")
    fp2 = fingerprint.get_stage_fingerprint(mod.stage)

    assert fp1 != fp2, "Instance class change must cause cache miss"


def test_nonlocal_class_instance_tracked(module_dir: pathlib.Path) -> None:
    """Class instances captured as nonlocals have their class tracked."""
    helpers_py = module_dir / "test_nonlocal_helpers.py"
    helpers_py.write_text(
        "class Processor:\n"
        + "    def process(self, x):\n"
        + "        return x * 2\n\n"
        + "def make_stage():\n"
        + "    processor = Processor()\n"
        + "    def stage():\n"
        + "        return processor.process(10)\n"
        + "    return stage\n\n"
        + "stage = make_stage()\n"
    )

    mod = _import_fresh("test_nonlocal_helpers")
    fp = fingerprint.get_stage_fingerprint(mod.stage)

    assert "class:processor.__class__" in fp, "Should capture nonlocal instance's class"
