"""Tests for change detection via fingerprinting.

These tests verify that the fingerprinting system correctly captures
transitive dependencies, constants, aliased functions, and module attributes.
"""

import inspect
import types
from typing import Any

from pivot import ast_utils, fingerprint


def helper_leaf(x: int) -> int:
    return x * 2


def helper_middle(x: int) -> int:
    return helper_leaf(x) + 1


def helper_top(x: int) -> int:
    return helper_middle(x) + 10


GLOBAL_CONSTANT = 42


def stage_direct_call(data: int) -> int:
    return helper_top(data)


def stage_with_constant(data: int) -> int:
    return data + GLOBAL_CONSTANT


def stage_with_alias(data: int) -> int:
    f = helper_top
    return f(data)


def test_direct_call_captures_transitive() -> None:
    """Direct function calls capture transitive dependencies."""
    cv = inspect.getclosurevars(stage_direct_call)
    assert "helper_top" in cv.globals, "Should see direct dependency"

    manifest = fingerprint.get_stage_fingerprint(stage_direct_call)

    assert "self:stage_direct_call" in manifest
    assert "func:helper_top" in manifest
    assert "func:helper_middle" in manifest
    assert "func:helper_leaf" in manifest


def test_global_constant_captured() -> None:
    """Global constants are captured in fingerprint."""
    cv = inspect.getclosurevars(stage_with_constant)
    assert "GLOBAL_CONSTANT" in cv.globals

    manifest = fingerprint.get_stage_fingerprint(stage_with_constant)

    assert "const:GLOBAL_CONSTANT" in manifest
    assert manifest["const:GLOBAL_CONSTANT"] == "42"


def test_aliasing_works() -> None:
    """Aliased functions (f = func; f()) are captured."""
    cv = inspect.getclosurevars(stage_with_alias)
    assert "helper_top" in cv.globals

    manifest = fingerprint.get_stage_fingerprint(stage_with_alias)

    assert "func:helper_top" in manifest
    assert "func:helper_middle" in manifest
    assert "func:helper_leaf" in manifest


def test_module_attr_google_style() -> None:
    """Google-style imports (import module; module.attr) are detected."""
    import json

    def stage_with_module(data: Any) -> str:
        return json.dumps(data)

    cv = inspect.getclosurevars(stage_with_module)
    # json is in nonlocals because it's captured from enclosing scope
    assert "json" in cv.nonlocals
    assert isinstance(cv.nonlocals["json"], types.ModuleType)

    attrs = ast_utils.extract_module_attr_usage(stage_with_module)
    assert ("json", "dumps") in attrs
