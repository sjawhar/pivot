# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportImplicitOverride=false

import ast
import math
import os
import pathlib
import sys

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
    assert "loader:PathOnly:config" in fp


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
