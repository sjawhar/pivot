"""Tests for Google-style imports with user modules.

Verifies that fingerprinting correctly captures module.attr usage patterns
like `user_utils.helper_b()` and `user_utils.CONSTANT_A`.
"""

import json as json_module

import tests.user_utils as user_utils
from pivot import ast_utils, fingerprint


def stage_google_style(data: int) -> int:
    """Stage function using Google-style imports."""
    return user_utils.helper_b(data) + user_utils.CONSTANT_A


def test_google_style_captures_module_attrs() -> None:
    """get_stage_fingerprint captures module.attr usage from user modules."""
    manifest = fingerprint.get_stage_fingerprint(stage_google_style)

    # Should capture the stage function itself
    assert "self:stage_google_style" in manifest, "Should capture stage function"

    # Should capture module attribute usage via AST analysis
    assert "mod:user_utils.helper_b" in manifest, "Should capture user_utils.helper_b"
    assert "mod:user_utils.CONSTANT_A" in manifest, "Should capture user_utils.CONSTANT_A"

    # Verify helper_b is hashed (16-char hex string) - user code callables are now properly hashed
    helper_b_hash = manifest["mod:user_utils.helper_b"]
    assert len(helper_b_hash) == 16, "helper_b should be a 16-char hash"
    assert all(c in "0123456789abcdef" for c in helper_b_hash), "helper_b hash should be hex"

    # Should also capture transitive dependencies via recursive fingerprinting
    assert "func:helper_a" in manifest, "Should capture transitive dep helper_a"
    assert "func:leaf_func" in manifest, "Should capture transitive dep leaf_func"


def test_extract_module_attr_usage_finds_patterns() -> None:
    """extract_module_attr_usage finds module.attr patterns in function AST."""
    attrs = ast_utils.extract_module_attr_usage(stage_google_style)

    # Should find both user_utils.helper_b and user_utils.CONSTANT_A
    assert ("user_utils", "helper_b") in attrs, "Should find user_utils.helper_b"
    assert ("user_utils", "CONSTANT_A") in attrs, "Should find user_utils.CONSTANT_A"


def test_is_user_code_identifies_user_modules() -> None:
    """is_user_code correctly identifies user module functions."""
    # user_utils functions should be identified as user code
    assert fingerprint.is_user_code(user_utils.helper_b), "helper_b should be user code"
    assert fingerprint.is_user_code(user_utils.helper_a), "helper_a should be user code"

    # stdlib functions should NOT be user code
    import json

    assert not fingerprint.is_user_code(json.dumps), "json.dumps should not be user code"


def test_unused_module_attrs_not_captured() -> None:
    """Fingerprint only captures actually used module attrs, not all exports."""
    manifest = fingerprint.get_stage_fingerprint(stage_google_style)

    # unused_func is defined in user_utils but not used by stage_google_style
    manifest_str = str(manifest)
    assert "unused_func" not in manifest_str, "unused_func should not be in manifest"


def test_google_style_captures_user_code_hash() -> None:
    """User-code callables via Google-style import are properly hashed.

    This tests the fix where user-code callables accessed via module.attr
    are properly hashed (not just marked as "callable").
    """
    # Get initial fingerprint
    manifest1 = fingerprint.get_stage_fingerprint(stage_google_style)
    hash1 = fingerprint.hash_function_ast(user_utils.helper_b)

    # Verify the hash is captured in the manifest
    assert manifest1["mod:user_utils.helper_b"] == hash1, "helper_b hash should be in manifest"

    # Verify that a different function would have a different hash
    hash_different = fingerprint.hash_function_ast(user_utils.helper_a)
    assert hash1 != hash_different, "Different functions should have different hashes"

    # Verify that the same function always produces the same hash (stability)
    hash1_again = fingerprint.hash_function_ast(user_utils.helper_b)
    assert hash1 == hash1_again, "Same function should produce same hash"


def _stage_using_json() -> str:
    """Module-level stage function using stdlib json module."""
    return json_module.dumps({"key": "value"})


def test_google_style_stdlib_callable_not_hashed() -> None:
    """Stdlib callables via Google-style import are marked 'callable', not hashed.

    This ensures we don't try to hash stdlib functions (which could be unstable
    across Python versions) and only hash user-defined code.
    """
    manifest = fingerprint.get_stage_fingerprint(_stage_using_json)

    # json.dumps is stdlib, so it should be marked as "callable" not hashed
    assert manifest.get("mod:json_module.dumps") == "callable", (
        "stdlib callable should be 'callable'"
    )
