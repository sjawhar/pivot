# pyright: reportUnusedFunction=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false
"""Tests for fingerprinting functools.partial and functools.wraps."""

import functools

from pivot import fingerprint

# --- Test helpers for functools.partial ---


def _helper_func(a: int, b: int, c: int = 10) -> int:
    """Helper function to be wrapped with partial."""
    return a + b + c


def _helper_func_v2(a: int, b: int, c: int = 10) -> int:
    """Different helper function with same signature."""
    return a * b + c


# --- Test helpers for functools.wraps ---


def _caching_decorator_v1(func):  # type: ignore[no-untyped-def]
    """Decorator version 1."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        print("CACHE_V1")  # Decorator-specific logic
        return func(*args, **kwargs)

    return wrapper


def _caching_decorator_v2(func):  # type: ignore[no-untyped-def]
    """Decorator version 2 - different logic."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        print("CACHE_V2")  # Different decorator logic!
        return func(*args, **kwargs)

    return wrapper


def _original_stage_func() -> int:
    """Original function to be decorated."""
    return 42


def _original_stage_func_v2() -> int:
    """Different original function."""
    return 99


# --- Tests for functools.partial ---


def test_partial_is_detected():
    """functools.partial objects are detected and tracked."""
    bound = functools.partial(_helper_func, 1, c=20)

    def stage_with_partial() -> int:
        return bound(2)

    fp = fingerprint.get_stage_fingerprint(stage_with_partial)

    # Should have partial:bound.args and partial:bound.kwargs
    assert "partial:bound.args" in fp, f"partial args not tracked. Got: {list(fp.keys())}"
    assert "partial:bound.kwargs" in fp, f"partial kwargs not tracked. Got: {list(fp.keys())}"
    # Should also track the underlying function
    assert "func:bound.func" in fp, f"partial underlying func not tracked. Got: {list(fp.keys())}"


def test_partial_args_change_triggers_fingerprint_change():
    """Changing bound args changes the fingerprint."""
    bound_v1 = functools.partial(_helper_func, 1)
    bound_v2 = functools.partial(_helper_func, 999)  # Different bound arg

    def stage_v1() -> int:
        return bound_v1(2, 3)

    def stage_v2() -> int:
        return bound_v2(2, 3)

    fp1 = fingerprint.get_stage_fingerprint(stage_v1)
    fp2 = fingerprint.get_stage_fingerprint(stage_v2)

    assert fp1["partial:bound_v1.args"] != fp2["partial:bound_v2.args"], (
        "Different bound args should produce different hashes"
    )


def test_partial_kwargs_change_triggers_fingerprint_change():
    """Changing bound kwargs changes the fingerprint."""
    bound_v1 = functools.partial(_helper_func, c=10)
    bound_v2 = functools.partial(_helper_func, c=999)  # Different bound kwarg

    def stage_v1() -> int:
        return bound_v1(1, 2)

    def stage_v2() -> int:
        return bound_v2(1, 2)

    fp1 = fingerprint.get_stage_fingerprint(stage_v1)
    fp2 = fingerprint.get_stage_fingerprint(stage_v2)

    assert fp1["partial:bound_v1.kwargs"] != fp2["partial:bound_v2.kwargs"], (
        "Different bound kwargs should produce different hashes"
    )


def test_partial_underlying_func_change_triggers_fingerprint_change():
    """Changing the underlying function changes the fingerprint."""
    bound_v1 = functools.partial(_helper_func, 1)
    bound_v2 = functools.partial(_helper_func_v2, 1)  # Different underlying func

    def stage_v1() -> int:
        return bound_v1(2, 3)

    def stage_v2() -> int:
        return bound_v2(2, 3)

    fp1 = fingerprint.get_stage_fingerprint(stage_v1)
    fp2 = fingerprint.get_stage_fingerprint(stage_v2)

    assert fp1["func:bound_v1.func"] != fp2["func:bound_v2.func"], (
        "Different underlying functions should produce different hashes"
    )


def test_partial_in_closure():
    """functools.partial in closure (nonlocals) is tracked."""

    def make_stage():  # type: ignore[no-untyped-def]
        bound = functools.partial(_helper_func, 1, c=20)

        def stage() -> int:
            return bound(2)

        return stage

    stage = make_stage()
    fp = fingerprint.get_stage_fingerprint(stage)

    # Should track partial from nonlocals
    assert "partial:bound.args" in fp, (
        f"partial args from closure not tracked. Got: {list(fp.keys())}"
    )


# --- Tests for functools.wraps ---


def test_wrapped_function_detected():
    """Functions decorated with functools.wraps are properly tracked."""

    @_caching_decorator_v1
    def my_stage() -> int:
        return 42

    fp = fingerprint.get_stage_fingerprint(my_stage)

    # The function should be tracked
    assert "self:my_stage" in fp, f"wrapped function not tracked. Got: {list(fp.keys())}"

    # The original function should also be tracked via closure
    assert "func:func" in fp, f"original function not tracked via closure. Got: {list(fp.keys())}"


def test_decorator_change_triggers_fingerprint_change():
    """Changing the decorator logic changes the fingerprint."""

    @_caching_decorator_v1
    def stage_v1() -> int:
        return 42

    @_caching_decorator_v2
    def stage_v2() -> int:
        return 42

    fp1 = fingerprint.get_stage_fingerprint(stage_v1)
    fp2 = fingerprint.get_stage_fingerprint(stage_v2)

    # The wrapper hashes should be different because decorator logic is different
    assert fp1["self:stage_v1"] != fp2["self:stage_v2"], (
        "Different decorator logic should produce different wrapper hashes"
    )


def test_original_function_change_triggers_fingerprint_change():
    """Changing the original (wrapped) function changes the fingerprint."""

    @_caching_decorator_v1
    def stage_v1() -> int:
        return 42

    @_caching_decorator_v1
    def stage_v2() -> int:
        return 99  # Different return value

    fp1 = fingerprint.get_stage_fingerprint(stage_v1)
    fp2 = fingerprint.get_stage_fingerprint(stage_v2)

    # The original function is tracked via closure as 'func'
    # The hashes should be different because the original functions differ
    assert fp1["func:func"] != fp2["func:func"], (
        "Different original functions should produce different hashes"
    )


def test_wrapped_uses_bytecode_not_source():
    """Wrapped functions use bytecode hash, not source (which follows __wrapped__)."""

    @_caching_decorator_v1
    def my_stage() -> int:
        """This docstring is in the original function."""
        return 42

    # Get fingerprint - should use bytecode for wrapper
    fp = fingerprint.get_stage_fingerprint(my_stage)
    wrapper_hash = fp["self:my_stage"]

    # The hash should be the wrapper's bytecode, not the original's AST
    # We can verify by checking that changing only the decorator changes the hash
    @_caching_decorator_v2
    def my_stage_v2() -> int:
        """This docstring is in the original function."""
        return 42

    fp2 = fingerprint.get_stage_fingerprint(my_stage_v2)
    wrapper_hash_v2 = fp2["self:my_stage_v2"]

    # If we were using source (which follows __wrapped__), these would be the same
    # because the original functions have the same source
    assert wrapper_hash != wrapper_hash_v2, (
        "Wrapper hash should differ when decorator changes (proves bytecode is used)"
    )


def test_nested_wraps():
    """Multiple levels of functools.wraps are tracked."""

    def decorator_a(func):  # type: ignore[no-untyped-def]
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            print("A")
            return func(*args, **kwargs)

        return wrapper

    def decorator_b(func):  # type: ignore[no-untyped-def]
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            print("B")
            return func(*args, **kwargs)

        return wrapper

    @decorator_a
    @decorator_b
    def my_stage() -> int:
        return 42

    fp = fingerprint.get_stage_fingerprint(my_stage)

    # Should have the outer wrapper (decorator_a's wrapper)
    assert "self:my_stage" in fp, f"outer wrapper not tracked. Got: {list(fp.keys())}"

    # The closure should include decorator_b's wrapper as 'func'
    assert "func:func" in fp, f"inner wrapper not tracked via closure. Got: {list(fp.keys())}"
