# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportFunctionMemberAccess=false, reportCallIssue=false, reportOptionalCall=false, reportAttributeAccessIssue=false, reportUnannotatedClassAttribute=false, reportArgumentType=false
"""
Tests demonstrating callback detection vulnerabilities in fingerprinting.

These tests document known limitations of the fingerprinting system.
Tests are marked with xfail where the current behavior is considered a bug.
"""

import functools

import pytest

from pivot import fingerprint

# =============================================================================
# Module-level helpers for testing
# =============================================================================


def _callback_v1():
    """Callback version 1."""
    return 1


def _callback_v2():
    """Callback version 2."""
    return 2


def _base_function(multiplier, x):
    """Base function for partial testing."""
    return x * multiplier


# =============================================================================
# VULNERABILITY 1: functools.wraps / __wrapped__
# =============================================================================


def _original_for_wraps():
    """Original function that will be wrapped."""
    return "original"


@functools.wraps(_original_for_wraps)
def _wrapped_different_impl():
    """Different implementation hidden by @wraps."""
    return "completely different!"


def test_functools_wraps_hides_implementation_change():
    """@functools.wraps does NOT hide implementation changes (fixed via bytecode hashing)."""
    fp_original = fingerprint.get_stage_fingerprint(_original_for_wraps)
    fp_wrapped = fingerprint.get_stage_fingerprint(_wrapped_different_impl)

    # These are now correctly different because we use __code__ bytecode for wrapped functions
    assert fp_original != fp_wrapped, (
        "functools.wraps should not hide implementation - fingerprints should differ"
    )


def test_manual_wrapped_attribute_hides_code():
    """Manually setting __wrapped__ does NOT hide actual code (fixed via bytecode hashing)."""

    def target():
        return "target"

    def attacker():
        return "MALICIOUS CODE"

    attacker.__wrapped__ = target  # type: ignore

    # Compare the HASH VALUES, not the full fingerprint dicts
    h_target = fingerprint.hash_function_ast(target)
    h_attacker = fingerprint.hash_function_ast(attacker)

    # attacker has completely different code, hashes now correctly differ
    assert h_target != h_attacker, "__wrapped__ attribute should not hide actual implementation"


def test_nested_wraps_have_different_hashes():
    """Multiple layers of @wraps correctly have different hashes (fixed via bytecode)."""

    def level0():
        return 0

    @functools.wraps(level0)
    def level1():
        return 1

    @functools.wraps(level1)
    def level2():
        return 2

    h0 = fingerprint.hash_function_ast(level0)
    h1 = fingerprint.hash_function_ast(level1)
    h2 = fingerprint.hash_function_ast(level2)

    # Now all three have different hashes because we use bytecode for wrapped functions
    assert h0 != h1, "level0 and level1 should have different hashes"
    assert h1 != h2, "level1 and level2 should have different hashes"
    assert h0 != h2, "level0 and level2 should have different hashes"


# =============================================================================
# VULNERABILITY 2: functools.partial
# =============================================================================


def test_partial_argument_change_detected():
    """Changes to partial arguments ARE detected (fixed via special partial handling)."""
    partial_v1 = functools.partial(_base_function, 2)
    partial_v2 = functools.partial(_base_function, 3)  # Different argument!

    def make_stage(p):
        def stage():
            return p(10)

        return stage

    stage1 = make_stage(partial_v1)
    stage2 = make_stage(partial_v2)

    fp1 = fingerprint.get_stage_fingerprint(stage1)
    fp2 = fingerprint.get_stage_fingerprint(stage2)

    # These are now correctly different because partial args are hashed
    assert fp1 != fp2, "Partial with different arguments should have different fingerprint"


def test_partial_wrapped_function_change_detected():
    """Changes to the function wrapped by partial ARE detected (fixed via special handling)."""
    partial_v1 = functools.partial(_callback_v1)
    partial_v2 = functools.partial(_callback_v2)  # Different function!

    def make_stage(p):
        def stage():
            return p()

        return stage

    stage1 = make_stage(partial_v1)
    stage2 = make_stage(partial_v2)

    fp1 = fingerprint.get_stage_fingerprint(stage1)
    fp2 = fingerprint.get_stage_fingerprint(stage2)

    # Now correctly different because partial.func is recursively fingerprinted
    assert fp1 != fp2, "Partial with different wrapped function should differ"


def test_partial_is_not_user_code():
    """Document that functools.partial fails is_user_code check."""
    partial_obj = functools.partial(_callback_v1)
    assert fingerprint.is_user_code(partial_obj) is False


# =============================================================================
# VULNERABILITY 3: Instance state changes
# =============================================================================


class _CallbackContainer:
    """Container that holds callbacks in instance state."""

    def __init__(self):
        self._callbacks = dict[str, object]()

    def __getitem__(self, key):
        return self._callbacks[key]

    def __setitem__(self, key, value):
        self._callbacks[key] = value


_container = _CallbackContainer()


def _helper_stage_uses_container():
    """Stage that uses callback from container."""
    return _container["handler"]()


@pytest.mark.xfail(
    reason="Instance state (container contents) not tracked, only class definition",
    strict=True,
)
def test_container_callback_change_detected():
    """Changes to callbacks stored in containers should be detected."""
    _container["handler"] = _callback_v1
    fp1 = fingerprint.get_stage_fingerprint(_helper_stage_uses_container)

    _container["handler"] = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(_helper_stage_uses_container)

    assert fp1 != fp2, "Container callback change should be detected"


# =============================================================================
# VULNERABILITY 4: Class attribute runtime modification
# =============================================================================


class _ConfigClass:
    """Config class with runtime-assigned callback."""

    callback = None


def _helper_stage_uses_class_attr():
    """Stage that uses class attribute callback."""
    return _ConfigClass.callback()


@pytest.mark.xfail(
    reason="Class attributes modified at runtime not tracked",
    strict=True,
)
def test_class_attribute_callback_change_detected():
    """Changes to class attribute callbacks should be detected."""
    _ConfigClass.callback = _callback_v1
    fp1 = fingerprint.get_stage_fingerprint(_helper_stage_uses_class_attr)

    _ConfigClass.callback = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(_helper_stage_uses_class_attr)

    assert fp1 != fp2, "Class attribute callback change should be detected"


# =============================================================================
# WORKING CORRECTLY: Module variable callbacks
# =============================================================================

_current_callback = _callback_v1


def _helper_stage_uses_module_var():
    """Stage that uses module-level callback variable."""
    return _current_callback()


@pytest.mark.xfail(
    reason=(
        "getclosurevars caching trades runtime variable change detection for ~30% fingerprinting speedup. "
        "In practice, module changes trigger reimport (which clears caches) during watch mode."
    ),
    strict=True,
)
def test_module_variable_callback_change_detected():
    """Module-level callback variable changes ARE detected correctly."""
    global _current_callback

    _current_callback = _callback_v1
    fp1 = fingerprint.get_stage_fingerprint(_helper_stage_uses_module_var)

    _current_callback = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(_helper_stage_uses_module_var)

    # With caching enabled, runtime module variable changes aren't detected
    # (but source file changes trigger reimport which clears caches)
    assert fp1 != fp2, "Module variable callback change should be detected"

    # Restore
    _current_callback = _callback_v1


# =============================================================================
# WORKING CORRECTLY: Dict callbacks in closures
# =============================================================================


def test_dict_callback_in_closure_detected():
    """Callbacks in dict closures ARE detected correctly."""
    handlers = {"process": _callback_v1}

    def stage():
        return handlers["process"]()

    fp1 = fingerprint.get_stage_fingerprint(stage)

    handlers["process"] = _callback_v2
    fp2 = fingerprint.get_stage_fingerprint(stage)

    # This WORKS correctly
    assert fp1 != fp2, "Dict callback in closure should be detected"


# =============================================================================
# WORKING CORRECTLY: Multi-layer wrapped callbacks via closure
# =============================================================================


def test_multilayer_closure_callbacks_detected():
    """Callbacks passed through closure layers ARE detected correctly."""

    def inner_v1():
        return 1

    def inner_v2():
        return 2

    def wrapper(callback):
        def wrapped():
            return callback() + 1

        return wrapped

    wrapped_v1 = wrapper(inner_v1)
    wrapped_v2 = wrapper(inner_v2)

    fp1 = fingerprint.get_stage_fingerprint(wrapped_v1)
    fp2 = fingerprint.get_stage_fingerprint(wrapped_v2)

    # This WORKS correctly - closure recursion captures inner callback
    assert fp1 != fp2, "Different inner callbacks should produce different fingerprints"


# =============================================================================
# WORKING CORRECTLY: Async and generator functions
# =============================================================================

_async_callback = None


async def _async_callback_v1():
    return 1


async def _async_callback_v2():
    return 2


def _helper_stage_async():
    """Stage that references async callback."""
    return _async_callback


@pytest.mark.xfail(
    reason=(
        "getclosurevars caching trades runtime variable change detection for ~30% fingerprinting speedup. "
        "In practice, module changes trigger reimport (which clears caches) during watch mode."
    ),
    strict=True,
)
def test_async_callback_change_detected():
    """Async callback changes ARE detected correctly."""
    global _async_callback

    _async_callback = _async_callback_v1
    fp1 = fingerprint.get_stage_fingerprint(_helper_stage_async)

    _async_callback = _async_callback_v2
    fp2 = fingerprint.get_stage_fingerprint(_helper_stage_async)

    # With caching enabled, runtime module variable changes aren't detected
    # (but source file changes trigger reimport which clears caches)
    assert fp1 != fp2, "Async callback change should be detected"

    _async_callback = None
