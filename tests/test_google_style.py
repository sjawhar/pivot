"""Tests for Google-style imports with user modules.

Verifies that fingerprinting correctly captures module.attr usage patterns
like `user_utils.helper_b()` and `user_utils.CONSTANT_A`.
"""

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

    # Verify helper_b is marked as callable
    assert manifest["mod:user_utils.helper_b"] == "callable", "helper_b should be marked callable"


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
