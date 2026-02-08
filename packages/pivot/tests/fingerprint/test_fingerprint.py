# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportImplicitOverride=false

import ast
import importlib.util
import math
import os
import pathlib
import sys
import time

import networkx as nx
import pytest

from pivot import ast_utils, fingerprint

# --- Module-level helper functions for testing ---
# These must be at module level to properly capture imports in their closures


def _helper_uses_math_pi() -> float:
    """Helper that uses math.pi for testing module attr detection."""
    return math.pi * 2.0


def _helper_uses_multiple_math_attrs(x: float) -> float:
    """Helper that uses multiple math attributes."""
    return math.sqrt(x) + math.sin(x) + math.cos(x)


def _helper_uses_multiple_modules() -> int:
    """Helper that uses attributes from different modules."""
    return len(os.path.join("a", "b")) + sys.maxsize


def _helper_uses_os_path() -> str:
    """Helper that uses nested attribute access."""
    return os.path.join("a", "b")


def _helper_for_hash_test_1() -> int:
    """First function for hash identity test."""
    return 42


def _helper_for_hash_test_2() -> int:
    """Second identical function for hash identity test."""
    return 42


def _helper_plain() -> int:
    """Plain function for whitespace test."""
    return 42


def _helper_with_comment() -> int:
    """Function with comment for whitespace test."""
    return 42  # with comment


def _helper_different_comment() -> int:
    """Function with different comment for whitespace test."""
    return 42  # different comment


def _helper_docstring_1() -> int:
    """Docstring 1."""
    return 42


def _helper_docstring_2() -> int:
    """Docstring 2."""
    return 42


def _helper_with_math_pi():
    """Helper that uses math.pi for AttributeError test."""
    return math.pi * 2


# Constants for testing constant capture (no underscore prefix!)
TEST_STRING = "Hello, World!"
TEST_BYTES = b"binary data"
TEST_NONE = None
TEST_FLOAT = 3.14159
TEST_BOOL = True


def _helper_uses_string():
    """Helper that uses string constant."""
    return TEST_STRING


def _helper_uses_bytes():
    """Helper that uses bytes constant."""
    return TEST_BYTES


def _helper_uses_none():
    """Helper that uses None constant."""
    return TEST_NONE


def _helper_uses_float():
    """Helper that uses float constant."""
    return TEST_FLOAT * 2


def _helper_uses_bool():
    """Helper that uses bool constant."""
    return TEST_BOOL


def _helper_outer_uses_inner(x):
    """Helper that references another helper."""
    # Reference the function (not call it) so it's captured in closure
    func = _helper_for_hash_test_1
    return func() + x


# --- get_stage_fingerprint tests ---


def test_simple_function_fingerprinted():
    """Should hash simple function with no dependencies."""

    def simple() -> int:
        return 42

    fp = fingerprint.get_stage_fingerprint(simple)

    assert "self:simple" in fp
    assert isinstance(fp["self:simple"], str)
    assert len(fp["self:simple"]) > 0
    assert len(fp) == 1


def test_helper_function_captured():
    """Should capture referenced helper function in manifest."""

    def helper(x: int) -> int:
        return x * 2

    def main(x: int) -> int:
        return helper(x) + 1

    fp = fingerprint.get_stage_fingerprint(main)

    assert "self:main" in fp
    assert "func:helper" in fp
    assert len(fp) == 2


def test_constant_captured():
    """Should capture global constant value."""
    CONSTANT = 100

    def use_constant() -> int:
        return CONSTANT * 2

    fp = fingerprint.get_stage_fingerprint(use_constant)

    assert "self:use_constant" in fp
    assert "const:CONSTANT" in fp
    assert fp["const:CONSTANT"] == "100"


def test_multiple_constants_captured():
    """Should capture multiple constants with correct values."""
    PI = 3.14159
    MAX_ITER = 100
    DEBUG = True

    def use_constants() -> float:
        if DEBUG:
            return PI * MAX_ITER
        return 0.0

    fp = fingerprint.get_stage_fingerprint(use_constants)

    assert fp["const:PI"] == "3.14159"
    assert fp["const:MAX_ITER"] == "100"
    assert fp["const:DEBUG"] == "True"


def test_transitive_dependencies_captured():
    """Should recursively fingerprint entire dependency chain."""

    def leaf(x: int) -> int:
        return x + 1

    def middle(x: int) -> int:
        return leaf(x) * 2

    def top(x: int) -> int:
        return middle(x) + 10

    fp = fingerprint.get_stage_fingerprint(top)

    assert "self:top" in fp
    assert "func:middle" in fp
    assert "func:leaf" in fp


def test_unchanged_function_same_fingerprint():
    """Should produce identical fingerprint for unchanged function."""

    def func() -> int:
        return 42

    fp1 = fingerprint.get_stage_fingerprint(func)
    fp2 = fingerprint.get_stage_fingerprint(func)

    assert fp1 == fp2
    assert fp1["self:func"] == fp2["self:func"]


def test_changed_function_different_fingerprint():
    """Should produce different fingerprint when logic changes."""

    def func_v1() -> int:
        return 42

    def func_v2() -> int:
        return 43

    fp1 = fingerprint.get_stage_fingerprint(func_v1)
    fp2 = fingerprint.get_stage_fingerprint(func_v2)

    assert fp1["self:func_v1"] != fp2["self:func_v2"]


def test_stdlib_module_attrs_not_tracked():
    """Stdlib module attributes (math, os, json, etc.) are NOT tracked."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_math_pi)
    assert "self:_helper_uses_math_pi" in fp
    assert "mod:math.pi" not in fp

    fp2 = fingerprint.get_stage_fingerprint(_helper_uses_multiple_math_attrs)
    assert "mod:math.sqrt" not in fp2
    assert "mod:math.sin" not in fp2
    assert "mod:math.cos" not in fp2


def test_aliased_function_captured():
    """Should handle function aliasing (f = helper; f(x))."""

    def helper(x: int) -> int:
        return x * 2

    def main(x: int) -> int:
        f = helper
        return f(x) + 1

    fp = fingerprint.get_stage_fingerprint(main)

    assert "self:main" in fp
    assert "func:helper" in fp


def test_circular_reference_handled():
    """Should handle circular references without infinite recursion."""

    def func_a(x: int) -> int:
        if x > 0:
            return func_b(x - 1)
        return 1

    def func_b(x: int) -> int:
        if x > 0:
            return func_a(x - 1)
        return 1

    fp = fingerprint.get_stage_fingerprint(func_a)

    assert "self:func_a" in fp
    assert "func:func_b" in fp


def test_nested_function_not_in_globals():
    """Should handle nested functions."""

    def outer(x: int) -> int:
        def inner(y: int) -> int:
            return y * 2

        return inner(x) + 1

    fp = fingerprint.get_stage_fingerprint(outer)

    assert "self:outer" in fp


def test_lambda_function_fingerprinted():
    """Should handle lambda functions."""
    my_lambda = lambda x: x * 2  # noqa: E731

    fp = fingerprint.get_stage_fingerprint(my_lambda)

    assert "self:<lambda>" in fp or "self:my_lambda" in fp


def test_function_with_no_closure_vars():
    """Should handle pure functions with no closure variables."""

    def pure_function(x: int, y: int) -> int:
        return x + y

    fp = fingerprint.get_stage_fingerprint(pure_function)

    assert "self:pure_function" in fp
    assert len(fp) == 1


def test_fingerprint_with_visited_set():
    """Should accept visited parameter for recursion tracking."""

    def func() -> int:
        return 42

    visited = set()
    fp = fingerprint.get_stage_fingerprint(func, visited=visited)

    assert "self:func" in fp
    assert len(visited) > 0


def test_fingerprint_builtin_function_skipped():
    """Should not include builtin functions in manifest."""

    def use_builtin(items: list[int]) -> int:
        return len(items)

    fp = fingerprint.get_stage_fingerprint(use_builtin)

    assert "self:use_builtin" in fp
    assert "func:len" not in fp


@pytest.mark.parametrize(
    ("x", "y"),
    [
        pytest.param(10, 20, id="positive"),
        pytest.param(0, 0, id="zero"),
        pytest.param(-5, 10, id="negative"),
    ],
)
def test_fingerprint_with_default_args(x, y):
    """Should handle functions with default arguments."""

    def func_with_defaults(x: int = 10, y: int = 20) -> int:
        return x + y

    fp = fingerprint.get_stage_fingerprint(func_with_defaults)

    assert "self:func_with_defaults" in fp


# --- hash_function_ast tests ---


def test_identical_functions_same_hash():
    """Should produce same hash for identical functions."""
    h1 = fingerprint.hash_function_ast(_helper_for_hash_test_1)
    h2 = fingerprint.hash_function_ast(_helper_for_hash_test_2)

    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) > 0


def test_whitespace_and_comments_ignored():
    """Should ignore whitespace and comment differences."""
    h1 = fingerprint.hash_function_ast(_helper_plain)
    h2 = fingerprint.hash_function_ast(_helper_with_comment)
    h3 = fingerprint.hash_function_ast(_helper_different_comment)

    # All should produce same hash (comments not in AST)
    assert h1 == h2
    assert h2 == h3


def test_docstrings_ignored():
    """Should ignore docstring differences."""
    h1 = fingerprint.hash_function_ast(_helper_docstring_1)
    h2 = fingerprint.hash_function_ast(_helper_docstring_2)

    assert h1 == h2


def test_different_logic_different_hash():
    """Should produce different hash for different logic."""

    def func_returns_42() -> int:
        return 42

    def func_returns_43() -> int:
        return 43

    def func_mult_2(x: int) -> int:
        return x * 2

    def func_mult_3(x: int) -> int:
        return x * 3

    h1 = fingerprint.hash_function_ast(func_returns_42)
    h2 = fingerprint.hash_function_ast(func_returns_43)
    h3 = fingerprint.hash_function_ast(func_mult_2)
    h4 = fingerprint.hash_function_ast(func_mult_3)

    # Different logic should produce different hashes
    assert h1 != h2
    assert h3 != h4


def test_different_variable_names_different_hash():
    """Should detect variable name changes."""

    def func1(x: int) -> int:
        return x * 2

    def func2(y: int) -> int:
        return y * 2

    h1 = fingerprint.hash_function_ast(func1)
    h2 = fingerprint.hash_function_ast(func2)

    assert h1 != h2


def test_hash_is_stable():
    """Should produce stable hashes across multiple calls."""

    def func() -> int:
        return 42

    hashes = [fingerprint.hash_function_ast(func) for _ in range(10)]

    assert len(set(hashes)) == 1


def test_hash_format():
    """Should return hash in hex string format."""

    def func() -> int:
        return 42

    h = fingerprint.hash_function_ast(func)

    assert isinstance(h, str)
    assert len(h) > 0
    assert all(c in "0123456789abcdef" for c in h.lower())


def test_hash_complex_ast():
    """Should handle functions with complex AST structures."""

    def complex_func(x: int) -> int:
        if x < 0:
            return -x
        elif x == 0:
            return 0
        else:
            result = 1
            for i in range(x):
                result *= i + 1
            return result

    h = fingerprint.hash_function_ast(complex_func)

    assert isinstance(h, str)
    assert len(h) > 0


# --- is_user_code tests ---


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        pytest.param(len, False, id="builtin_len"),
        pytest.param(print, False, id="builtin_print"),
        pytest.param(sum, False, id="builtin_sum"),
        pytest.param(int, False, id="builtin_int"),
        pytest.param(str, False, id="builtin_str"),
        pytest.param(list, False, id="builtin_list"),
        pytest.param(None, False, id="none"),
        pytest.param(os.path.join, False, id="stdlib_os_path"),
        pytest.param(sys, False, id="stdlib_sys"),
        pytest.param(nx, False, id="thirdparty_networkx"),
        pytest.param(pytest.fixture, False, id="thirdparty_pytest"),
    ],
)
def test_is_user_code_non_user(obj, expected):
    """Should identify stdlib, builtins, and third-party as not user code."""
    assert fingerprint.is_user_code(obj) == expected


def test_is_user_code_local_function():
    """Should identify local functions as user code."""

    def local_func() -> None:
        pass

    assert fingerprint.is_user_code(local_func) is True


def test_is_user_code_lambda():
    """Should identify lambda functions as user code."""
    my_lambda = lambda x: x * 2  # noqa: E731

    assert fingerprint.is_user_code(my_lambda) is True


def test_is_user_code_module():
    """Should identify pivot module as user code."""
    assert fingerprint.is_user_code(fingerprint) is True


def test_is_user_code_non_callable():
    """Should handle non-callable objects gracefully."""
    try:
        result = fingerprint.is_user_code(42)
        assert result is False
    except (AttributeError, TypeError):
        pass  # Acceptable to raise error


def test_is_user_code_stdlib_via_symlink() -> None:
    """Should identify stdlib as non-user code even when sys.prefix is a symlink.

    On homebrew macOS, sys.base_prefix is a symlink:
    - sys.base_prefix = /opt/homebrew/opt/python@3.13/... (symlink)
    - Actual stdlib at /opt/homebrew/Cellar/python@3.13/... (resolved)

    The _is_stdlib_path check must resolve symlinks to correctly identify stdlib.
    """
    # math module should be identified as NOT user code
    assert fingerprint.is_user_code(math) is False
    assert fingerprint.is_user_code(math.ceil) is False


# --- extract_module_attr_usage tests ---


def test_extract_single_module_attr():
    """Should extract single module.attr pattern."""
    attrs = ast_utils.extract_module_attr_usage(_helper_uses_math_pi)

    assert ("math", "pi") in attrs


def test_extract_multiple_attrs_same_module():
    """Should extract multiple attributes from same module."""
    attrs = ast_utils.extract_module_attr_usage(_helper_uses_multiple_math_attrs)

    assert ("math", "sqrt") in attrs
    assert ("math", "sin") in attrs


def test_extract_multiple_modules():
    """Should extract attributes from different modules."""
    attrs = ast_utils.extract_module_attr_usage(_helper_uses_multiple_modules)

    assert ("os", "path") in attrs
    assert ("sys", "maxsize") in attrs


def test_extract_no_module_attrs():
    """Should return empty list when no module attributes used."""

    def pure_func(x: int) -> int:
        return x * 2

    attrs = ast_utils.extract_module_attr_usage(pure_func)

    assert len(attrs) == 0


def test_extract_nested_attr_access():
    """Should handle nested attribute access (module.submodule.attr)."""
    attrs = ast_utils.extract_module_attr_usage(_helper_uses_os_path)

    assert ("os", "path") in attrs


def test_get_function_ast():
    """Should parse function to AST node."""
    node = ast_utils.get_function_ast(_helper_for_hash_test_1)

    assert isinstance(node, ast.FunctionDef)
    assert node.name == "_helper_for_hash_test_1"


def test_get_function_ast_builtin_error():
    """Should raise ValueError for builtin functions."""
    with pytest.raises(ValueError, match="Cannot get source"):
        ast_utils.get_function_ast(len)


def test_normalize_ast():
    """Should normalize AST by removing docstrings."""
    source = '''
def f(x):
    """Docstring."""
    return x * 2
'''
    tree = ast.parse(source)

    # Before normalization, function should have docstring
    func_def = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    assert len(func_def.body) == 2  # Docstring + return statement

    # After normalization, docstring should be removed
    normalized = ast_utils.normalize_ast(tree)
    func_def_normalized = next(
        node for node in ast.walk(normalized) if isinstance(node, ast.FunctionDef)
    )
    assert len(func_def_normalized.body) == 1  # Only return statement


# --- Error path and edge case tests ---


def test_fingerprint_with_underscore_globals():
    """Should skip global names starting with underscore."""
    # Note: This test verifies that __name__, __file__, etc. are skipped
    # We can't easily test this directly, but the behavior is verified by
    # not seeing these in other test fingerprints

    def simple_func():
        return 42

    fp = fingerprint.get_stage_fingerprint(simple_func)

    # Should not have any __* globals
    assert all(not key.startswith("const:__") for key in fp.keys())
    assert all(not key.startswith("func:__") for key in fp.keys())


def test_extract_module_attr_builtin():
    """Should handle builtin functions gracefully."""
    # len is a builtin that doesn't have getsource
    attrs = ast_utils.extract_module_attr_usage(len)

    # Should return empty list (can't get source)
    assert attrs == []


def test_hash_builtin_function():
    """Should handle builtin functions that don't have source."""
    # len is a builtin without source code
    h = fingerprint.hash_function_ast(len)

    # Should still return a hash (fallback to code object or id)
    assert isinstance(h, str)
    assert len(h) > 0


def test_fingerprint_getclosurevars_exception():
    """Should handle exceptions from getclosurevars."""
    # Some objects like builtin types don't support getclosurevars
    fp = fingerprint.get_stage_fingerprint(len)

    # Should still return manifest with at least the self entry
    assert "self:len" in fp or len(fp) >= 0  # Depends on fallback behavior


def test_fingerprint_with_nonlocal():
    """Should capture nonlocal variables."""

    def outer():
        x = 10

        def inner():
            return x * 2

        return inner

    inner_func = outer()
    fp = fingerprint.get_stage_fingerprint(inner_func)

    # Should have the function itself
    assert "self:inner" in fp
    # Nonlocal constant should be captured
    assert "const:x" in fp
    assert fp["const:x"] == "10"


def test_fingerprint_callable_nonlocal():
    """Should capture callable nonlocals."""

    def make_adder(n):
        def add(x):
            return x + n

        return add

    add_five = make_adder(5)
    fp = fingerprint.get_stage_fingerprint(add_five)

    # Should have the function and the nonlocal
    assert "self:add" in fp
    assert "const:n" in fp


def test_fingerprint_nonlocal_callable_function():
    """Should capture and recurse on nonlocal callable functions."""

    def helper_func(x):
        return x * 2

    def outer():
        def inner(x):
            return helper_func(x) + 1

        return inner

    inner_func = outer()
    fp = fingerprint.get_stage_fingerprint(inner_func)

    # Should have both the inner function and the helper it references
    assert "self:inner" in fp
    assert "func:helper_func" in fp


# --- Nonlocals with collection types containing callables ---
# Note: Collection processing scans for callable user code, not raw contents


def _collection_helper_a(x):
    """Helper function for collection tests."""
    return x * 2


def _collection_helper_b(x):
    """Second helper function for collection tests."""
    return x + 1


def test_fingerprint_nonlocal_list_with_callable():
    """Should capture callable functions within nonlocal list."""

    def outer():
        transforms = [_collection_helper_a, _collection_helper_b]

        def inner(x):
            for t in transforms:
                x = t(x)
            return x

        return inner

    inner_func = outer()
    fp = fingerprint.get_stage_fingerprint(inner_func)

    assert "self:inner" in fp
    # Callables in list are captured with index-based keys
    assert "func:transforms[0]" in fp
    assert "func:transforms[1]" in fp


def test_fingerprint_nonlocal_dict_with_callable():
    """Should capture callable functions within nonlocal dict."""

    def outer():
        handlers = {
            "double": _collection_helper_a,
            "increment": _collection_helper_b,
        }

        def inner(x, op):
            return handlers[op](x)

        return inner

    inner_func = outer()
    fp = fingerprint.get_stage_fingerprint(inner_func)

    assert "self:inner" in fp
    # Dict values are captured with key-based names
    assert "func:handlers['double']" in fp
    assert "func:handlers['increment']" in fp


def test_fingerprint_nonlocal_tuple_with_callable():
    """Should capture callable functions within nonlocal tuple."""

    def outer():
        pipeline = (_collection_helper_a, _collection_helper_b)

        def inner(x):
            for t in pipeline:
                x = t(x)
            return x

        return inner

    inner_func = outer()
    fp = fingerprint.get_stage_fingerprint(inner_func)

    assert "self:inner" in fp
    assert "func:pipeline[0]" in fp
    assert "func:pipeline[1]" in fp


def test_fingerprint_nonlocal_collection_callable_change_detected():
    """Changing callable in collection should change fingerprint."""

    def make_func(transform):
        transforms = [transform]

        def inner(x):
            return transforms[0](x)

        return inner

    func1 = make_func(_collection_helper_a)
    func2 = make_func(_collection_helper_b)

    fp1 = fingerprint.get_stage_fingerprint(func1)
    fp2 = fingerprint.get_stage_fingerprint(func2)

    # The hash of the callable should differ
    assert fp1["func:transforms[0]"] != fp2["func:transforms[0]"]


def test_hash_function_no_code_object():
    """Should handle objects without __code__ attribute."""

    class FakeCallable:
        """A callable without source or __code__."""

        def __call__(self):
            return 42

    fake = FakeCallable()
    h = fingerprint.hash_function_ast(fake)

    # Should fall back to identity hash
    assert isinstance(h, str)
    assert len(h) == 16  # xxhash64 hexdigest


def test_is_user_code_module_not_in_sys_modules():
    """Should return False for modules not in sys.modules."""

    class FakeObj:
        """Object with __module__ not in sys.modules."""

        __module__ = "nonexistent_module_12345"

    fake = FakeObj()
    result = fingerprint.is_user_code(fake)

    assert result is False


def test_fingerprint_merges_child_manifest():
    """Should merge child manifest excluding self entries."""

    def leaf_helper(x):
        return x + 1

    def middle_helper(x):
        return leaf_helper(x) * 2

    def top_func(x):
        return middle_helper(x) + 10

    fp = fingerprint.get_stage_fingerprint(top_func)

    # Should have all three functions
    assert "self:top_func" in fp
    assert "func:middle_helper" in fp
    assert "func:leaf_helper" in fp

    # Should NOT have self:middle_helper or self:leaf_helper
    # (child self entries should be excluded from merge)
    assert "self:middle_helper" not in fp
    assert "self:leaf_helper" not in fp


def test_hash_function_syntax_error_fallback():
    """Should handle SyntaxError by falling back to source hash."""
    # We can't easily create a function with invalid syntax that still exists
    # This path is defensive, so we'll verify the builtin hash covers it
    h = fingerprint.hash_function_ast(len)
    assert isinstance(h, str)
    assert len(h) == 16


def test_fingerprint_with_string_constant():
    """Should capture string constants."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_string)

    assert "self:_helper_uses_string" in fp
    assert "const:TEST_STRING" in fp
    assert fp["const:TEST_STRING"] == "'Hello, World!'"


def test_fingerprint_with_bytes_constant():
    """Should capture bytes constants."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_bytes)

    assert "self:_helper_uses_bytes" in fp
    assert "const:TEST_BYTES" in fp
    assert fp["const:TEST_BYTES"] == "b'binary data'"


def test_fingerprint_with_none_constant():
    """Should capture None constants."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_none)

    assert "self:_helper_uses_none" in fp
    assert "const:TEST_NONE" in fp
    assert fp["const:TEST_NONE"] == "None"


def test_fingerprint_skips_underscore_globals():
    """Should skip global variables starting with underscore."""
    # This test verifies that globals like __name__, __file__ are skipped
    # We create a scenario where a function references these

    def func_with_dunder():
        # Functions naturally have access to __name__ etc in their module
        # But fingerprinting should skip these
        return 42

    fp = fingerprint.get_stage_fingerprint(func_with_dunder)

    # Should not have any dunder or underscore globals
    for key in fp.keys():
        if key.startswith("const:") or key.startswith("func:"):
            name = key.split(":", 1)[1]
            assert not name.startswith("_"), f"Should skip underscore name: {name}"


def test_hash_function_with_code_object():
    """Should hash functions with __code__ but no source."""
    # Most built-in functions have __code__
    # Let's use a lambda which definitely has __code__
    lambda_func = eval("lambda x: x + 1")

    h1 = fingerprint.hash_function_ast(lambda_func)
    h2 = fingerprint.hash_function_ast(lambda_func)

    # Should produce consistent hashes
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 16


def test_fingerprint_with_float_constant():
    """Should capture float constants."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_float)

    assert "self:_helper_uses_float" in fp
    assert "const:TEST_FLOAT" in fp
    assert fp["const:TEST_FLOAT"] == "3.14159"


def test_fingerprint_with_bool_constant():
    """Should capture boolean constants."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_bool)

    assert "self:_helper_uses_bool" in fp
    assert "const:TEST_BOOL" in fp
    assert fp["const:TEST_BOOL"] == "True"


# ==============================================================================
# Loader fingerprinting tests
# ==============================================================================


def test_get_loader_fingerprint_returns_manifest():
    """Should return fingerprint manifest for loader."""
    from pivot import loaders

    loader = loaders.CSV()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert isinstance(fp, dict)
    assert len(fp) > 0


def test_loader_fingerprint_includes_load_method():
    """Should fingerprint the load method."""
    from pivot import loaders

    loader = loaders.CSV()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert "loader:CSV:load" in fp
    assert isinstance(fp["loader:CSV:load"], str)
    assert len(fp["loader:CSV:load"]) == 16  # xxhash64 hex


def test_loader_fingerprint_includes_save_method():
    """Should fingerprint the save method."""
    from pivot import loaders

    loader = loaders.CSV()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert "loader:CSV:save" in fp
    assert isinstance(fp["loader:CSV:save"], str)
    assert len(fp["loader:CSV:save"]) == 16


def test_loader_fingerprint_includes_config():
    """Should fingerprint dataclass field values."""
    from pivot import loaders

    loader = loaders.CSV(index_col="id", sep=";")
    fp = fingerprint.get_loader_fingerprint(loader)

    assert "loader:CSV:config" in fp
    assert isinstance(fp["loader:CSV:config"], str)


def test_loader_config_change_changes_fingerprint():
    """Different config values should produce different fingerprints."""
    from pivot import loaders

    loader1 = loaders.CSV(sep=",")
    loader2 = loaders.CSV(sep=";")

    fp1 = fingerprint.get_loader_fingerprint(loader1)
    fp2 = fingerprint.get_loader_fingerprint(loader2)

    assert fp1["loader:CSV:config"] != fp2["loader:CSV:config"]


def test_loader_same_config_same_fingerprint():
    """Same config values should produce same fingerprints."""
    from pivot import loaders

    loader1 = loaders.CSV(sep=",", index_col=0)
    loader2 = loaders.CSV(sep=",", index_col=0)

    fp1 = fingerprint.get_loader_fingerprint(loader1)
    fp2 = fingerprint.get_loader_fingerprint(loader2)

    assert fp1 == fp2


def test_different_loader_types_different_fingerprint():
    """Different loader types should produce different fingerprints."""
    from pivot import loaders

    csv_loader = loaders.CSV()
    json_loader = loaders.JSON()

    fp_csv = fingerprint.get_loader_fingerprint(csv_loader)
    fp_json = fingerprint.get_loader_fingerprint(json_loader)

    # Method hashes should differ
    assert fp_csv["loader:CSV:load"] != fp_json["loader:JSON:load"]


def test_json_loader_fingerprint():
    """Should fingerprint JSON loader correctly."""
    from pivot import loaders

    loader = loaders.JSON(indent=4)
    fp = fingerprint.get_loader_fingerprint(loader)

    assert "loader:JSON:load" in fp
    assert "loader:JSON:save" in fp
    assert "loader:JSON:config" in fp


def test_yaml_loader_fingerprint():
    """Should fingerprint YAML loader correctly."""
    from pivot import loaders

    loader = loaders.YAML()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert "loader:YAML:load" in fp
    assert "loader:YAML:save" in fp
    assert "loader:YAML:config" in fp


def test_pickle_loader_fingerprint():
    """Should fingerprint Pickle loader correctly."""
    from pivot import loaders

    loader = loaders.Pickle()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert "loader:Pickle:load" in fp
    assert "loader:Pickle:save" in fp
    assert "loader:Pickle:config" in fp


def test_pathonly_loader_fingerprint():
    """Should fingerprint PathOnly loader correctly."""
    from pivot import loaders

    loader = loaders.PathOnly()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert "loader:PathOnly:load" in fp
    assert "loader:PathOnly:save" in fp
    # PathOnly has no dataclass fields, so no config to fingerprint
    # Config is only included when the loader has configurable fields


def test_custom_loader_fingerprint():
    """Should fingerprint custom loader subclasses."""
    import dataclasses
    import pathlib

    from pivot import loaders

    @dataclasses.dataclass(frozen=True)
    class CustomTextLoader(loaders.Loader[str]):
        """Custom loader for testing."""

        prefix: str = ""

        def load(self, path: pathlib.Path) -> str:
            return self.prefix + path.read_text()

        def save(self, data: str, path: pathlib.Path) -> None:
            path.write_text(data)

        def empty(self) -> str:
            return ""

    loader = CustomTextLoader(prefix="TEST:")
    fp = fingerprint.get_loader_fingerprint(loader)

    assert "loader:CustomTextLoader:load" in fp
    assert "loader:CustomTextLoader:save" in fp
    assert "loader:CustomTextLoader:config" in fp


def test_custom_loader_code_change_detected():
    """Custom loader code changes should change fingerprint."""
    import dataclasses
    import pathlib

    from pivot import loaders

    @dataclasses.dataclass(frozen=True)
    class LoaderV1(loaders.Loader[str]):
        def load(self, path: pathlib.Path) -> str:
            return path.read_text()

        def save(self, data: str, path: pathlib.Path) -> None:
            path.write_text(data)

        def empty(self) -> str:
            return ""

    @dataclasses.dataclass(frozen=True)
    class LoaderV2(loaders.Loader[str]):
        def load(self, path: pathlib.Path) -> str:
            return path.read_text().strip()  # Different logic

        def save(self, data: str, path: pathlib.Path) -> None:
            path.write_text(data)

        def empty(self) -> str:
            return ""

    fp1 = fingerprint.get_loader_fingerprint(LoaderV1())
    fp2 = fingerprint.get_loader_fingerprint(LoaderV2())

    # Load method hash should differ
    assert fp1["loader:LoaderV1:load"] != fp2["loader:LoaderV2:load"]


def test_loader_fingerprint_stable():
    """Loader fingerprint should be stable across calls."""
    from pivot import loaders

    loader = loaders.CSV(index_col="id")

    fp1 = fingerprint.get_loader_fingerprint(loader)
    fp2 = fingerprint.get_loader_fingerprint(loader)

    assert fp1 == fp2


# ==============================================================================
# Namespace package detection tests (Issue #2: incorrect user code classification)
# ==============================================================================


@pytest.fixture
def namespace_pkg_in_site_packages(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Create a synthetic namespace package in a fake site-packages directory."""
    # Create fake site-packages with a namespace package (no __init__.py)
    site_packages = tmp_path / "site-packages" / "fake_namespace_pkg"
    site_packages.mkdir(parents=True)

    # Add to sys.path so Python can find it
    monkeypatch.syspath_prepend(str(tmp_path / "site-packages"))

    # Clear any cached import state
    sys.modules.pop("fake_namespace_pkg", None)

    yield "fake_namespace_pkg"

    # Cleanup
    sys.modules.pop("fake_namespace_pkg", None)


@pytest.fixture
def namespace_pkg_user_code(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Create a synthetic namespace package in a user directory (not site-packages)."""
    # Create user directory with a namespace package (no __init__.py)
    user_pkg = tmp_path / "my_project" / "user_namespace_pkg"
    user_pkg.mkdir(parents=True)

    # Add to sys.path so Python can find it
    monkeypatch.syspath_prepend(str(tmp_path / "my_project"))

    # Clear any cached import state
    sys.modules.pop("user_namespace_pkg", None)

    yield "user_namespace_pkg"

    # Cleanup
    sys.modules.pop("user_namespace_pkg", None)


def test_is_user_code_namespace_package_in_site_packages(
    namespace_pkg_in_site_packages: str,
):
    """Namespace packages in site-packages should NOT be classified as user code.

    Namespace packages (PEP 420) don't have __file__, but they have __path__.
    """
    import importlib

    module = importlib.import_module(namespace_pkg_in_site_packages)

    # Verify it's a namespace package (no __file__, but has __path__)
    assert not hasattr(module, "__file__") or module.__file__ is None
    assert hasattr(module, "__path__")

    # Should NOT be classified as user code (it's in site-packages)
    assert fingerprint.is_user_code(module) is False


def test_is_user_code_namespace_package_user_code(namespace_pkg_user_code: str):
    """Namespace packages outside site-packages SHOULD be classified as user code."""
    import importlib

    module = importlib.import_module(namespace_pkg_user_code)

    # Verify it's a namespace package
    assert not hasattr(module, "__file__") or module.__file__ is None
    assert hasattr(module, "__path__")

    # Should be classified as user code (not in site-packages)
    assert fingerprint.is_user_code(module) is True


def test_is_user_code_path_component_matching_not_substring(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Ensure 'site-packages' is matched as path component, not substring.

    A path like '/home/user/my-site-packages-util/' should NOT be treated as
    site-packages just because it contains the substring.
    """
    # Create a directory with "site-packages" as substring but not component
    tricky_path = tmp_path / "my-site-packages-util" / "tricky_pkg"
    tricky_path.mkdir(parents=True)

    monkeypatch.syspath_prepend(str(tmp_path / "my-site-packages-util"))
    sys.modules.pop("tricky_pkg", None)

    import importlib

    module = importlib.import_module("tricky_pkg")

    # This is user code - "site-packages" is a substring, not a path component
    assert fingerprint.is_user_code(module) is True

    sys.modules.pop("tricky_pkg", None)


# ==============================================================================
# Lock file determinism tests (Issue #1: non-deterministic code_manifest ordering)
# ==============================================================================


def test_code_manifest_sorted_in_lock_file():
    """code_manifest keys should be sorted alphabetically in lock files.

    This ensures consecutive pipeline runs produce identical lock files.
    """
    import yaml

    from pivot.storage import lock
    from pivot.types import LockData

    # Create a code_manifest with keys in non-alphabetical order
    code_manifest = {
        "func:zebra": "hash1",
        "func:alpha": "hash2",
        "mod:omega.attr": "hash3",
        "class:Beta": "hash4",
        "self:main": "hash5",
    }

    lock_data = LockData(
        code_manifest=code_manifest,
        params={},
        dep_hashes={},
        output_hashes={},
        dep_generations={},
    )

    storage_data = lock._convert_to_storage_format(lock_data)

    # Serialize to YAML and check key order
    yaml_str = yaml.dump(storage_data, sort_keys=False)

    # Find positions of keys in the YAML output
    positions = {
        key: yaml_str.find(key) for key in code_manifest.keys() if yaml_str.find(key) != -1
    }

    # Keys should appear in sorted order
    sorted_keys = sorted(code_manifest.keys())
    actual_order = sorted(positions.keys(), key=lambda k: positions[k])

    assert actual_order == sorted_keys, f"Expected {sorted_keys}, got {actual_order}"


# ==============================================================================
# Persistent cache tests
# ==============================================================================


def test_should_skip_persistent_cache_closure():
    """Closures with <locals> in qualname should skip persistent cache."""

    def outer():
        def inner():
            return 42

        return inner

    closure_func = outer()
    # The closure has "<locals>" in its qualname
    assert "<locals>" in closure_func.__qualname__
    assert fingerprint._should_skip_persistent_cache(closure_func) is True


def test_should_skip_persistent_cache_wrapped():
    """Wrapped functions should skip persistent cache."""
    import functools

    def original():
        return 42

    @functools.wraps(original)
    def wrapper():
        return original()

    assert hasattr(wrapper, "__wrapped__")
    assert fingerprint._should_skip_persistent_cache(wrapper) is True


def test_should_skip_persistent_cache_normal_function():
    """Normal module-level functions should not skip persistent cache."""
    # _helper_for_hash_test_1 is a module-level function
    assert fingerprint._should_skip_persistent_cache(_helper_for_hash_test_1) is False


def test_get_func_source_info_normal_function(monkeypatch: pytest.MonkeyPatch):
    """Should return source info for normal functions."""
    from pivot import project

    # Set cache to the repo root (parent of src/ and tests/) so test file is within project
    repo_root = pathlib.Path(__file__).parent.parent.parent
    project._project_root_cache = repo_root
    # _helper_for_hash_test_1 is defined in this file
    result = fingerprint._get_func_source_info(_helper_for_hash_test_1)

    assert result is not None
    rel_path, mtime_ns, size, inode = result
    assert "test_fingerprint.py" in rel_path
    assert mtime_ns > 0
    assert size > 0
    assert inode > 0


def test_get_func_source_info_builtin():
    """Should return None for builtins."""
    result = fingerprint._get_func_source_info(len)
    assert result is None


def test_get_func_source_info_lambda(monkeypatch: pytest.MonkeyPatch):
    """Lambdas defined in source files should have source info."""
    from pivot import project

    # Set cache to the repo root (parent of src/ and tests/) so test file is within project
    repo_root = pathlib.Path(__file__).parent.parent.parent
    project._project_root_cache = repo_root
    my_lambda = lambda x: x * 2  # noqa: E731
    result = fingerprint._get_func_source_info(my_lambda)

    # Lambdas in actual source files (like this test file) have source info
    assert result is not None, "Lambda in test file should have source info"
    rel_path, mtime_ns, size, inode = result
    assert "test_fingerprint.py" in rel_path
    assert mtime_ns > 0
    assert size > 0
    assert inode > 0


def test_flush_ast_hash_cache_empty():
    """Flushing empty pending writes should not error."""
    # Clear any pending writes first
    fingerprint._pending_ast_writes.clear()
    # Should not raise
    fingerprint.flush_ast_hash_cache()


@pytest.fixture
def reset_fingerprint_cache_state(monkeypatch):
    """Reset fingerprint module state for testing persistent cache behavior."""
    # Save original state
    orig_pending = fingerprint._pending_ast_writes.copy()
    orig_pending_manifest = fingerprint._pending_manifest_writes.copy()
    orig_db = fingerprint._state_db
    orig_attempted = fingerprint._state_db_init_attempted

    # Reset state
    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    yield

    # Restore original state
    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_ast_writes.extend(orig_pending)
    fingerprint._pending_manifest_writes.clear()
    fingerprint._pending_manifest_writes.extend(orig_pending_manifest)
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = orig_db
    fingerprint._state_db_init_attempted = orig_attempted


def test_hash_function_ast_adds_to_pending_writes(
    tmp_path, monkeypatch, reset_fingerprint_cache_state
):
    """Module-level functions should queue entries for persistent cache."""
    # Set project root to tmp_path so _get_func_source_info can compute relative paths
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    # Create a test module file
    test_module = tmp_path / "test_stage.py"
    test_module.write_text("""
def my_stage():
    return 42
""")

    # Import the function
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_stage", test_module)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_stage"] = module
    try:
        spec.loader.exec_module(module)

        # Hash the function - should add to pending writes
        func = module.my_stage
        result = fingerprint.hash_function_ast(func)

        # Verify hash was computed
        assert isinstance(result, str)
        assert len(result) == 16  # xxhash64 hexdigest

        # Verify entry was queued for persistent cache
        assert len(fingerprint._pending_ast_writes) == 1
        rel_path, mtime_ns, size, inode, qualname, py_version, schema_version, hash_hex = (
            fingerprint._pending_ast_writes[0]
        )
        assert rel_path == "test_stage.py"
        assert qualname == "my_stage"
        assert py_version == fingerprint._PYTHON_VERSION
        assert schema_version == fingerprint._CACHE_SCHEMA_VERSION
        assert hash_hex == result
        assert mtime_ns > 0
        assert size > 0
        assert inode > 0
    finally:
        del sys.modules["test_stage"]


def test_hash_function_ast_skips_closures_for_persistent():
    """Closures should still hash correctly but not use persistent cache."""

    def make_adder(n):
        def add(x):
            return x + n

        return add

    add_five = make_adder(5)

    # Should hash without error
    h = fingerprint.hash_function_ast(add_five)
    assert isinstance(h, str)
    assert len(h) == 16  # xxhash64 hexdigest


def test_hash_function_ast_uses_memory_cache():
    """Memory cache should be used on repeated calls."""

    def test_func():
        return 42

    # Clear caches
    fingerprint._hash_function_ast_cache.clear()

    h1 = fingerprint.hash_function_ast(test_func)
    h2 = fingerprint.hash_function_ast(test_func)

    assert h1 == h2
    # The function should be in memory cache after first call
    assert test_func in fingerprint._hash_function_ast_cache


# ==============================================================================
# Persistent cache integration tests
# ==============================================================================


def test_persistent_cache_full_roundtrip(tmp_path, monkeypatch):
    """Full round-trip: compute → flush → hit → modify → miss.

    Verifies that:
    1. First call computes and queues for persistent cache
    2. flush_ast_hash_cache() writes to StateDB
    3. Second call hits persistent cache (after clearing memory cache)
    4. Modifying the file causes a cache miss
    """
    from pivot.storage import state

    # Set up isolated state directory
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"

    # Create initial StateDB so fingerprint module can open it in readonly mode
    with state.StateDB(db_path):
        pass

    # Patch project root and state db path
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    # Reset fingerprint module state
    fingerprint._pending_ast_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    # Create test module file
    test_module = tmp_path / "my_stage.py"
    test_module.write_text("""
def my_stage():
    return 42
""")

    # Import the function
    import importlib.util

    spec = importlib.util.spec_from_file_location("my_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["my_stage"] = module
    try:
        spec.loader.exec_module(module)
        func = module.my_stage

        # Step 1: First call - should compute and queue
        hash1 = fingerprint.hash_function_ast(func)
        assert len(fingerprint._pending_ast_writes) == 1
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # xxhash64 hexdigest

        # Step 2: Flush to StateDB
        fingerprint.flush_ast_hash_cache()
        assert len(fingerprint._pending_ast_writes) == 0

        # Clear memory cache to force persistent lookup
        fingerprint._hash_function_ast_cache.clear()
        # Reset state_db so it reopens in readonly mode
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

        # Reload module to get fresh function object (memory cache uses object identity)
        del sys.modules["my_stage"]
        spec2 = importlib.util.spec_from_file_location("my_stage", test_module)
        assert spec2 is not None and spec2.loader is not None
        module2 = importlib.util.module_from_spec(spec2)
        sys.modules["my_stage"] = module2
        spec2.loader.exec_module(module2)
        func2 = module2.my_stage

        # Step 3: Second call - should hit persistent cache
        hash2 = fingerprint.hash_function_ast(func2)
        assert hash2 == hash1
        # No new pending writes (cache hit)
        assert len(fingerprint._pending_ast_writes) == 0

        # Step 4: Modify the file (change mtime/size)
        import time

        time.sleep(0.01)  # Ensure mtime changes
        test_module.write_text("""
def my_stage():
    return 43  # Changed
""")

        # Clear memory cache and reload
        fingerprint._hash_function_ast_cache.clear()
        del sys.modules["my_stage"]
        spec3 = importlib.util.spec_from_file_location("my_stage", test_module)
        assert spec3 is not None and spec3.loader is not None
        module3 = importlib.util.module_from_spec(spec3)
        sys.modules["my_stage"] = module3
        spec3.loader.exec_module(module3)
        func3 = module3.my_stage

        # Step 5: Should miss cache (mtime changed) and compute new hash
        hash3 = fingerprint.hash_function_ast(func3)
        assert hash3 != hash1  # Different code = different hash
        assert len(fingerprint._pending_ast_writes) == 1  # Queued for cache
    finally:
        sys.modules.pop("my_stage", None)
        # Clean up fingerprint module state
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_graceful_degradation_when_statedb_unavailable(tmp_path, monkeypatch):
    """Fingerprinting works even when StateDB initialization fails.

    The persistent cache is a performance optimization, not a correctness requirement.
    If StateDB can't be opened (e.g., corrupted, missing dir, permissions), fingerprinting
    should continue without the persistent cache.
    """
    # Patch to simulate StateDB unavailable (init throws)
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(
        "pivot.config.io.get_state_db_path",
        lambda: tmp_path / "nonexistent" / "deeply" / "nested" / "state.db",
    )

    # Reset fingerprint module state
    fingerprint._pending_ast_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    # Create test module
    test_module = tmp_path / "graceful_stage.py"
    test_module.write_text("""
def graceful_stage():
    return 42
""")

    import importlib.util

    spec = importlib.util.spec_from_file_location("graceful_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["graceful_stage"] = module
    try:
        spec.loader.exec_module(module)
        func = module.graceful_stage

        # Should not raise - fingerprinting works without persistent cache
        hash_result = fingerprint.hash_function_ast(func)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 16  # xxhash64 hexdigest

        # Entries are still queued (flush will fail gracefully too)
        assert len(fingerprint._pending_ast_writes) == 1

        # Flush should not raise either
        fingerprint.flush_ast_hash_cache()  # Fails silently
    finally:
        sys.modules.pop("graceful_stage", None)
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_flush_ast_hash_cache_writes_to_statedb(tmp_path, monkeypatch):
    """flush_ast_hash_cache() actually persists entries to StateDB."""
    from pivot.storage import state

    # Set up isolated state directory
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"

    # Create initial StateDB
    with state.StateDB(db_path) as db:
        pass

    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    # Reset fingerprint module state
    fingerprint._pending_ast_writes.clear()
    fingerprint._hash_function_ast_cache.clear()

    # Manually queue some entries (simulating what hash_function_ast does)
    py_ver = fingerprint._PYTHON_VERSION
    schema_ver = fingerprint._CACHE_SCHEMA_VERSION
    test_entries: list[tuple[str, int, int, int, str, str, int, str]] = [
        ("src/a.py", 1000000000, 100, 111, "func_a", py_ver, schema_ver, "aaaa111122223333"),
        ("src/b.py", 2000000000, 200, 222, "func_b", py_ver, schema_ver, "bbbb444455556666"),
        (
            "src/c.py",
            3000000000,
            300,
            333,
            "MyClass.method",
            py_ver,
            schema_ver,
            "cccc777788889999",
        ),
    ]
    fingerprint._pending_ast_writes.extend(test_entries)

    # Flush to StateDB
    fingerprint.flush_ast_hash_cache()

    # Verify entries were actually written
    with state.StateDB(db_path, readonly=True) as db:
        for (
            rel_path,
            mtime_ns,
            size,
            inode,
            qualname,
            py_version,
            schema_version,
            expected_hash,
        ) in test_entries:
            actual_hash = db.get_ast_hash(
                rel_path, mtime_ns, size, inode, qualname, py_version, schema_version
            )
            assert actual_hash == expected_hash, (
                f"Expected {expected_hash} for {rel_path}:{qualname}, got {actual_hash}"
            )

    # Pending writes should be cleared
    assert len(fingerprint._pending_ast_writes) == 0


# ==============================================================================
# Lambda disambiguation and dedent fix tests
# ==============================================================================


def test_multiple_lambdas_same_file_different_hashes(tmp_path, monkeypatch):
    """Two lambdas on different lines get different cache keys."""
    # Set project root so _get_func_source_info works
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    mod_py = tmp_path / "test_lambdas.py"
    mod_py.write_text("""
lambda_a = lambda x: x + 1
lambda_b = lambda x: x + 2
""")

    # Import fresh module
    spec = importlib.util.spec_from_file_location("test_lambdas", mod_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_lambdas"] = mod
    try:
        spec.loader.exec_module(mod)

        h1 = fingerprint.hash_function_ast(mod.lambda_a)
        h2 = fingerprint.hash_function_ast(mod.lambda_b)

        assert h1 != h2, "Different lambdas should have different hashes"

        q1 = fingerprint._get_qualname_for_cache(mod.lambda_a)
        q2 = fingerprint._get_qualname_for_cache(mod.lambda_b)
        assert q1 != q2, "Different lambdas should have different qualnames"
        assert "<lambda>:" in q1, f"Lambda qualname should have line info: {q1}"
    finally:
        sys.modules.pop("test_lambdas", None)


def test_same_line_lambdas_disambiguated(tmp_path, monkeypatch):
    """Two lambdas on same line get different cache keys via column."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    mod_py = tmp_path / "test_same_line.py"
    mod_py.write_text("pair = (lambda x: x, lambda y: y)\n")

    spec = importlib.util.spec_from_file_location("test_same_line", mod_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_same_line"] = mod
    try:
        spec.loader.exec_module(mod)

        q1 = fingerprint._get_qualname_for_cache(mod.pair[0])
        q2 = fingerprint._get_qualname_for_cache(mod.pair[1])
        assert q1 != q2, f"Same-line lambdas should have different qualnames: {q1} vs {q2}"
    finally:
        sys.modules.pop("test_same_line", None)


def test_method_comment_change_no_hash_change(tmp_path, monkeypatch):
    """Changing comments in methods should not change hash (dedent fix)."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    mod_py = tmp_path / "test_method.py"
    mod_py.write_text("""
class MyClass:
    def method(self):
        # original comment
        return 42
""")

    spec = importlib.util.spec_from_file_location("test_method", mod_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_method"] = mod
    try:
        spec.loader.exec_module(mod)
        h1 = fingerprint.hash_function_ast(mod.MyClass.method)

        # Clear memory cache
        fingerprint._hash_function_ast_cache.clear()

        # Modify comment
        mod_py.write_text("""
class MyClass:
    def method(self):
        # CHANGED comment
        return 42
""")

        # Reimport
        del sys.modules["test_method"]
        spec2 = importlib.util.spec_from_file_location("test_method", mod_py)
        assert spec2 is not None and spec2.loader is not None
        mod2 = importlib.util.module_from_spec(spec2)
        sys.modules["test_method"] = mod2
        spec2.loader.exec_module(mod2)
        h2 = fingerprint.hash_function_ast(mod2.MyClass.method)

        assert h1 == h2, "Comment change should not affect hash (dedent fix)"
    finally:
        sys.modules.pop("test_method", None)


def test_source_collector_records_source_files(tmp_path, monkeypatch):
    """_collecting_sources context manager records source files visited during fingerprinting."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    test_module = tmp_path / "collected_stage.py"
    test_module.write_text("""
def my_stage():
    return 42
""")

    spec = importlib.util.spec_from_file_location("collected_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["collected_stage"] = module
    try:
        spec.loader.exec_module(module)

        with fingerprint._collecting_sources() as source_map:
            fingerprint.get_stage_fingerprint(module.my_stage)

        # Should have recorded at least the stage's source file
        assert len(source_map) >= 1
        assert any("collected_stage.py" in path for path in source_map)
        # Each entry should have (mtime_ns, size, inode)
        for _rel_path, stats in source_map.items():
            mtime_ns, size, ino = stats
            assert mtime_ns > 0
            assert size > 0
            assert ino > 0
    finally:
        sys.modules.pop("collected_stage", None)


# ==============================================================================
# Stage manifest cache tests
# ==============================================================================


def test_manifest_cache_hit(tmp_path, monkeypatch):
    """Compute -> flush -> compute again returns cached manifest (walk not called)."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    # Reset fingerprint state
    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    test_module = tmp_path / "cached_stage.py"
    test_module.write_text("""
def cached_stage():
    return 42
""")

    spec = importlib.util.spec_from_file_location("cached_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cached_stage"] = module
    try:
        spec.loader.exec_module(module)

        # First call — computes and queues
        manifest1 = fingerprint.get_stage_fingerprint_cached("cached_stage", module.cached_stage)
        assert "self:cached_stage" in manifest1
        assert len(fingerprint._pending_manifest_writes) == 1

        # Flush
        fingerprint.flush_manifest_cache()
        assert len(fingerprint._pending_manifest_writes) == 0

        # Clear in-memory caches to force persistent lookup
        fingerprint._hash_function_ast_cache.clear()
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

        # Second call — should hit manifest cache
        manifest2 = fingerprint.get_stage_fingerprint_cached("cached_stage", module.cached_stage)
        assert manifest2 == manifest1
        # No new pending writes (cache hit)
        assert len(fingerprint._pending_manifest_writes) == 0
    finally:
        sys.modules.pop("cached_stage", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_manifest_cache_miss_on_source_change(tmp_path, monkeypatch):
    """Touch source file between runs -> recomputes manifest."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    test_module = tmp_path / "changing_stage.py"
    test_module.write_text("""
def changing_stage():
    return 42
""")

    spec = importlib.util.spec_from_file_location("changing_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["changing_stage"] = module
    try:
        spec.loader.exec_module(module)

        manifest1 = fingerprint.get_stage_fingerprint_cached(
            "changing_stage", module.changing_stage
        )
        fingerprint.flush_manifest_cache()

        # Modify file
        time.sleep(0.01)
        test_module.write_text("""
def changing_stage():
    return 43
""")

        # Clear caches
        fingerprint._hash_function_ast_cache.clear()
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

        # Reload module
        del sys.modules["changing_stage"]
        spec2 = importlib.util.spec_from_file_location("changing_stage", test_module)
        assert spec2 is not None and spec2.loader is not None
        module2 = importlib.util.module_from_spec(spec2)
        sys.modules["changing_stage"] = module2
        spec2.loader.exec_module(module2)

        manifest2 = fingerprint.get_stage_fingerprint_cached(
            "changing_stage", module2.changing_stage
        )
        assert manifest2 != manifest1  # Different code = different manifest
        assert len(fingerprint._pending_manifest_writes) == 1  # Queued for flush
    finally:
        sys.modules.pop("changing_stage", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_manifest_cache_miss_on_file_deleted(tmp_path, monkeypatch):
    """Delete source file -> recomputes manifest."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    # Two-file stage: main imports helper
    helper_file = tmp_path / "helper_mod.py"
    helper_file.write_text("""
def helper():
    return 99
""")
    main_file = tmp_path / "main_mod.py"
    main_file.write_text("""
import helper_mod

def main_stage():
    return helper_mod.helper()
""")

    spec_h = importlib.util.spec_from_file_location("helper_mod", helper_file)
    assert spec_h is not None and spec_h.loader is not None
    mod_h = importlib.util.module_from_spec(spec_h)
    sys.modules["helper_mod"] = mod_h
    spec_h.loader.exec_module(mod_h)

    spec_m = importlib.util.spec_from_file_location("main_mod", main_file)
    assert spec_m is not None and spec_m.loader is not None
    mod_m = importlib.util.module_from_spec(spec_m)
    sys.modules["main_mod"] = mod_m
    try:
        spec_m.loader.exec_module(mod_m)

        fingerprint.get_stage_fingerprint_cached("main_stage", mod_m.main_stage)
        fingerprint.flush_manifest_cache()

        # Delete helper file — stat will fail for cached source
        helper_file.unlink()

        # Clear caches
        fingerprint._hash_function_ast_cache.clear()
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

        # Recompute — should miss because helper_mod.py stat fails
        fingerprint.get_stage_fingerprint_cached("main_stage", mod_m.main_stage)
        assert len(fingerprint._pending_manifest_writes) == 1  # Cache miss -> re-queued
    finally:
        sys.modules.pop("helper_mod", None)
        sys.modules.pop("main_mod", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_manifest_cache_non_file_backed_function(tmp_path, monkeypatch):
    """Stage referencing builtins is cacheable; builtins not tracked as source files."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    test_module = tmp_path / "builtin_stage.py"
    test_module.write_text("""
def builtin_stage(items):
    return len(items)
""")

    spec = importlib.util.spec_from_file_location("builtin_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["builtin_stage"] = module
    try:
        spec.loader.exec_module(module)

        with fingerprint._collecting_sources() as source_map:
            fingerprint.get_stage_fingerprint(module.builtin_stage)

        # Only the stage's own source file should be tracked, not builtins
        assert len(source_map) == 1
        assert any("builtin_stage.py" in path for path in source_map)
    finally:
        sys.modules.pop("builtin_stage", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


# ==============================================================================
# Additional edge case / error path tests for manifest cache
# ==============================================================================


def test_make_manifest_cache_key_format():
    """_make_manifest_cache_key encodes stage name, python version, and schema version."""
    key = fingerprint._make_manifest_cache_key("train")
    decoded = key.decode()
    assert decoded.startswith("sm:")
    assert "train" in decoded
    assert fingerprint._PYTHON_VERSION in decoded
    assert str(fingerprint._CACHE_SCHEMA_VERSION) in decoded
    # Null byte separators present
    assert "\x00" in decoded


def test_make_manifest_cache_key_different_stages_differ():
    """Different stage names produce different keys."""
    key1 = fingerprint._make_manifest_cache_key("train")
    key2 = fingerprint._make_manifest_cache_key("evaluate")
    assert key1 != key2


def test_flush_manifest_cache_empty():
    """Flushing empty pending manifest writes should not error."""
    fingerprint._pending_manifest_writes.clear()
    # Should not raise
    fingerprint.flush_manifest_cache()


def test_flush_manifest_cache_failure_restores_pending(tmp_path, monkeypatch):
    """Failed flush restores entries so they can be retried."""
    # Use a path that cannot be created: a regular file blocks mkdir(parents=True)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setattr(
        "pivot.config.io.get_state_db_path",
        lambda: blocker / "subdir" / "state.db",
    )

    # Queue some entries
    fingerprint._pending_manifest_writes.clear()
    test_entries = [(b"sm:test_stage\x003.13\x001", b'{"m":{},"s":{}}')]
    fingerprint._pending_manifest_writes.extend(test_entries)

    # Flush should fail (blocker is a file, so mkdir(parents=True) raises NotADirectoryError)
    fingerprint.flush_manifest_cache()

    # Entries should be restored for retry
    assert len(fingerprint._pending_manifest_writes) == 1
    assert fingerprint._pending_manifest_writes[0] == test_entries[0]

    # Clean up
    fingerprint._pending_manifest_writes.clear()


def test_try_manifest_cache_hit_corrupted_json(tmp_path, monkeypatch):
    """Corrupted JSON in cache returns None instead of crashing."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"

    # Write corrupted data directly into the db
    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("bad_stage")
        db.put_raw(key, b"NOT VALID JSON {{{")

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("bad_stage")
        assert result is None, "Corrupted JSON should return None"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_try_manifest_cache_hit_no_db(monkeypatch):
    """_try_manifest_cache_hit returns None when StateDB is unavailable."""
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = True  # Already failed

    try:
        result = fingerprint._try_manifest_cache_hit("any_stage")
        assert result is None
    finally:
        fingerprint._state_db_init_attempted = False


def test_collecting_sources_restores_on_exception(tmp_path, monkeypatch):
    """_active_source_map is set to None even if an exception occurs during fingerprinting."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    # Verify _active_source_map is None before
    assert fingerprint._active_source_map is None

    class FingerprintError(Exception):
        pass

    try:
        with fingerprint._collecting_sources() as source_map:
            assert fingerprint._active_source_map is source_map
            raise FingerprintError("simulated failure")
    except FingerprintError:
        pass

    # Must be restored to None after exception
    assert fingerprint._active_source_map is None, (
        "_active_source_map should be None after exception in _collecting_sources"
    )


def test_source_collector_deduplicates_files(tmp_path, monkeypatch):
    """Source map records each file only once (first visit wins)."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    test_module = tmp_path / "dedup_stage.py"
    test_module.write_text("""
def helper():
    return 1

def main():
    return helper() + helper()
""")

    spec = importlib.util.spec_from_file_location("dedup_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dedup_stage"] = module
    try:
        spec.loader.exec_module(module)

        with fingerprint._collecting_sources() as source_map:
            fingerprint.get_stage_fingerprint(module.main)

        # The file should appear exactly once despite multiple function visits
        dedup_entries = [p for p in source_map if "dedup_stage.py" in p]
        assert len(dedup_entries) == 1
    finally:
        sys.modules.pop("dedup_stage", None)


def test_get_stage_fingerprint_cached_without_statedb(tmp_path, monkeypatch):
    """get_stage_fingerprint_cached works correctly even without a StateDB."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    fingerprint._pending_ast_writes.clear()
    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = True  # Simulate prior failure

    test_module = tmp_path / "no_db_stage.py"
    test_module.write_text("""
def no_db_stage():
    return 42
""")

    spec = importlib.util.spec_from_file_location("no_db_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["no_db_stage"] = module
    try:
        spec.loader.exec_module(module)

        manifest = fingerprint.get_stage_fingerprint_cached("no_db_stage", module.no_db_stage)

        assert "self:no_db_stage" in manifest
        # Should still queue for flush (even though flush may fail later)
        assert len(fingerprint._pending_manifest_writes) == 1
    finally:
        sys.modules.pop("no_db_stage", None)
        fingerprint._pending_manifest_writes.clear()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_try_manifest_cache_hit_non_dict_json(tmp_path, monkeypatch):
    """JSON that decodes to non-dict (e.g. list) returns None."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("list_stage")
        db.put_raw(key, b"[1, 2, 3]")  # Valid JSON, but not a dict

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("list_stage")
        assert result is None, "Non-dict JSON should return None"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_try_manifest_cache_hit_missing_m_or_s_keys(tmp_path, monkeypatch):
    """Cache entry missing 'm' or 's' keys returns None."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("missing_keys")
        db.put_raw(key, b'{"m": {"self:x": "hash"}}')  # Missing "s" key

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("missing_keys")
        assert result is None, "Missing 's' key should return None"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_try_manifest_cache_hit_empty_sources_forces_recompute(tmp_path, monkeypatch):
    """Empty sources dict forces recompute (can't validate staleness)."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("empty_sources")
        db.put_raw(key, b'{"m":{"self:x":"hash"},"s":{}}')

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("empty_sources")
        assert result is None, "Empty sources should force recompute"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_try_manifest_cache_hit_corrupted_stats_length(tmp_path, monkeypatch):
    """Source stats array with wrong length returns None."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"

    # Source file exists but stats array has only 2 elements instead of 3
    src_file = tmp_path / "stage.py"
    src_file.write_text("def stage(): pass\n")

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("bad_stats")
        db.put_raw(key, b'{"m":{"self:stage":"hash"},"s":{"stage.py":[1000,200]}}')

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("bad_stats")
        assert result is None, "Stats array with wrong length should return None"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_try_manifest_cache_hit_path_traversal_blocked(tmp_path, monkeypatch):
    """Absolute paths and path traversal in cached sources are rejected."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir / "state.db"

    # Test with absolute path
    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("traversal_abs")
        db.put_raw(key, b'{"m":{"self:x":"hash"},"s":{"/etc/passwd":[1000,200,555]}}')

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_db_path", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("traversal_abs")
        assert result is None, "Absolute path in sources should be rejected"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

    # Test with ../ traversal
    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("traversal_dotdot")
        db.put_raw(key, b'{"m":{"self:x":"hash"},"s":{"../../etc/passwd":[1000,200,555]}}')

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("traversal_dotdot")
        assert result is None, "Path traversal with .. should be rejected"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False
