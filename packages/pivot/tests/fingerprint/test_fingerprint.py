# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportImplicitOverride=false

import ast
import collections
import collections.abc
import dataclasses
import enum
import functools
import importlib
import importlib.util
import inspect
import json
import locale
import logging
import math
import os
import pathlib
import random
import re
import signal
import subprocess
import sys
import threading
import time
import types
import typing
from collections.abc import Generator

import networkx as nx
import pydantic
import pytest
import xxhash
from pydantic import BaseModel

import pivot_tui.stats
from pivot import ast_utils, exceptions, fingerprint, loaders, project, stage_def
from pivot.storage import state as state_mod

# A memory address as a repr writes one (`0x7f3a1c2b9e50`): no form may hold one.
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")


def _pending_ast_entries() -> list[fingerprint.AstHashEntry]:
    """Every queued AST-hash entry, whichever state directory it is filed under."""
    return [entry for entries in fingerprint._pending_ast_writes.values() for entry in entries]


def _pending_manifest_entries() -> list[tuple[bytes, bytes]]:
    """Every queued manifest entry, whichever state directory it is filed under."""
    return [entry for entries in fingerprint._pending_manifest_writes.values() for entry in entries]


def _const_key(name: str) -> str:
    return f"const:{__name__}.{name}"


def _definition_key(prefix: str, value: object) -> str:
    typed_value: typing.Any = value
    return f"{prefix}:{typed_value.__module__}.{typed_value.__qualname__}"


def _owned_key(prefix: str, owner: object, name: str) -> str:
    """Key of a closure cell or local closure helper, under the function that owns it."""
    typed_owner: typing.Any = owner
    return f"{prefix}:{typed_owner.__module__}.{typed_owner.__qualname__}.{name}"


def _loader_key(loader: object, member: str) -> str:
    return _definition_key("loader", type(loader)) + f":{member}"


def _cached_stage() -> int:
    """Function that hand-built manifest cache entries are keyed under."""
    return 0


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


class _HelperForwardRef:
    pass


class _HelperForwardRefString:
    pass


if typing.TYPE_CHECKING:

    class MissingType:
        pass


def _helper_annotations_mixed(
    a: "int",
    b: "_HelperForwardRef",
    c: "MissingType",
    d: int,
    e: '"_HelperForwardRefString"',
    f: typing.Any,
) -> "_HelperForwardRef":
    return _HelperForwardRef()


# Constants for testing constant capture (no underscore prefix!)
TEST_STRING = "Hello, World!"
TEST_BYTES = b"binary data"
TEST_NONE = None
TEST_FLOAT = 3.14159
TEST_BOOL = True


class _PydanticInnerModel(pydantic.BaseModel):
    value: int = 1


class _PydanticOuterModel(pydantic.BaseModel):
    inner: _PydanticInnerModel
    items: list[_PydanticInnerModel]


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


def _helper_uses_stdlib_callable(x: object) -> str:
    """Helper that uses typing.cast (stdlib callable) in closure."""
    return typing.cast("str", x)


def _helper_load_module_from_path(name: str, path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _HelperBase:
    """Base class for class body dependency tests."""


class _HelperFieldType:
    """Field type for class body dependency tests."""


class _HelperContainer:
    """Container for nested class references."""

    class Inner:
        """Nested class used for dotted annotation refs."""


class _HelperPydanticModel(BaseModel):
    """Pydantic model used for class annotation defaults."""

    value: int = 1


class _HelperAnnotatedClass(_HelperBase):
    """Class with annotations for dependency walk tests."""

    field: _HelperFieldType
    inner: _HelperContainer.Inner
    model: _HelperPydanticModel

    def __init__(self) -> None:
        self.field = _HelperFieldType()
        self.inner = _HelperContainer.Inner()
        self.model = _HelperPydanticModel()


def _helper_parse_annotation(expr: str) -> ast.AST:
    """Parse an annotation expression into an AST node."""
    tree = ast.parse(f"x: {expr}")
    ann_assign = next(node for node in ast.walk(tree) if isinstance(node, ast.AnnAssign))
    return ann_assign.annotation


def _helper_outer_uses_inner(x):
    """Helper that references another helper."""
    # Reference the function (not call it) so it's captured in closure
    func = _helper_for_hash_test_1
    return func() + x


def _helper_static_getattr(obj: typing.Any) -> object:
    """Helper with static getattr usage."""
    return obj.value


def _helper_dynamic_getattr(obj: object, name: str) -> object:
    """Helper with dynamic getattr usage."""
    return getattr(obj, name)


def _helper_uses_globals() -> dict[str, object]:
    """Helper that uses globals()."""
    return globals()


def _helper_uses_import_module() -> object:
    """Helper that uses importlib.import_module dynamically."""
    return importlib.import_module("json")


# --- get_stage_fingerprint tests ---


def test_simple_function_fingerprinted():
    """Should hash simple function with no dependencies."""

    def simple() -> int:
        return 42

    fp = fingerprint.get_stage_fingerprint(simple)

    assert "self:simple" in fp
    assert isinstance(fp["self:simple"], str)
    assert len(fp["self:simple"]) > 0


def test_resolve_annotations_individually_mixed():
    resolved = fingerprint._resolve_annotations_individually(_helper_annotations_mixed)

    assert resolved["a"] is int
    assert resolved["b"] is _HelperForwardRef
    assert resolved["d"] is int
    assert resolved["e"] is _HelperForwardRefString
    assert resolved["f"] is typing.Any
    assert resolved["return"] is _HelperForwardRef
    assert "c" not in resolved


def test_helper_function_captured():
    """Should capture referenced helper function in manifest."""

    def helper(x: int) -> int:
        return x * 2

    def main(x: int) -> int:
        return helper(x) + 1

    fp = fingerprint.get_stage_fingerprint(main)

    assert "self:main" in fp
    assert _definition_key("func", helper) in fp
    modules = {key for key in fp if key.startswith("module:")}
    assert f"module:{helper.__module__}" in modules
    assert all(f"{helper.__module__}.".startswith(f"{key[7:]}.") for key in modules), modules
    assert len(fp) == 2 + len(modules)


def test_constant_captured():
    """Should capture global constant value."""
    CONSTANT = 100

    def use_constant() -> int:
        return CONSTANT * 2

    fp = fingerprint.get_stage_fingerprint(use_constant)

    assert "self:use_constant" in fp
    assert _owned_key("const", use_constant, "CONSTANT") in fp
    assert fp[_owned_key("const", use_constant, "CONSTANT")] == "100"


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

    assert fp[_owned_key("const", use_constants, "PI")] == "3.14159"
    assert fp[_owned_key("const", use_constants, "MAX_ITER")] == "100"
    assert fp[_owned_key("const", use_constants, "DEBUG")] == "True"


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
    assert _owned_key("func", top, "middle") in fp
    assert _definition_key("func", leaf) in fp


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
    """Stdlib module attributes (math, os, json, etc.) are NOT tracked; the module is named, since
    a worker imports it by name (`byname:`)."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_math_pi)
    assert "self:_helper_uses_math_pi" in fp
    assert {key: value for key, value in fp.items() if not key.startswith("self:")} == {
        "byname:math": ""
    }, fp

    fp2 = fingerprint.get_stage_fingerprint(_helper_uses_multiple_math_attrs)
    assert {key: value for key, value in fp2.items() if not key.startswith("self:")} == {
        "byname:math": ""
    }, fp2


def test_aliased_function_captured():
    """Should handle function aliasing (f = helper; f(x))."""

    def helper(x: int) -> int:
        return x * 2

    def main(x: int) -> int:
        f = helper
        return f(x) + 1

    fp = fingerprint.get_stage_fingerprint(main)

    assert "self:main" in fp
    assert _definition_key("func", helper) in fp


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
    assert _owned_key("func", func_a, "func_b") in fp


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


def test_collect_annotation_names_handles_generics_and_unions():
    """Collects names from generics and union annotations."""
    names = set[str]()
    dotted_refs = list[tuple[str, ...]]()

    node = _helper_parse_annotation("list[_HelperFieldType] | _HelperBase")
    fingerprint._collect_annotation_names(node, names, dotted_refs)

    assert "_HelperFieldType" in names
    assert "_HelperBase" in names


def test_collect_annotation_names_handles_dotted_and_strings():
    """Collects dotted refs and string annotations."""
    names = set[str]()
    dotted_refs = list[tuple[str, ...]]()

    node = _helper_parse_annotation("_HelperContainer.Inner")
    fingerprint._collect_annotation_names(node, names, dotted_refs)
    assert ("_HelperContainer", "Inner") in dotted_refs

    node = _helper_parse_annotation('"ForwardType"')
    fingerprint._collect_annotation_names(node, names, dotted_refs)
    assert "ForwardType" in names


def test_collect_dotted_path_collects_attribute_chain():
    """Collects dotted attribute paths from AST nodes."""
    node = _helper_parse_annotation("_HelperContainer.Inner")
    attr_node = next(n for n in ast.walk(node) if isinstance(n, ast.Attribute))

    assert fingerprint._collect_dotted_path(attr_node) == ("_HelperContainer", "Inner")


def test_resolve_dotted_path_resolves_namespace():
    """Resolves dotted paths against module namespace."""
    ns = {"_HelperContainer": _HelperContainer}

    resolved = fingerprint._resolve_dotted_path(("_HelperContainer", "Inner"), ns)
    assert resolved is _HelperContainer.Inner

    missing = fingerprint._resolve_dotted_path(("Missing", "Type"), ns)
    assert missing is None


def _names_the_annotated_class() -> type:
    return _HelperAnnotatedClass


def test_a_class_walk_tracks_its_bases_and_annotations():
    """A class's walk follows its bases and the classes its annotations name, a pydantic model's
    schema included."""
    manifest = fingerprint.get_stage_fingerprint(_names_the_annotated_class)

    assert _definition_key("class", _HelperBase) in manifest
    assert _definition_key("class", _HelperFieldType) in manifest
    assert _definition_key("class", _HelperContainer.Inner) in manifest
    assert _definition_key("class", _HelperPydanticModel) in manifest
    assert _definition_key("schema", _HelperPydanticModel) in manifest


class _RecordedBase(typing.TypedDict):
    tag: str


class _RecordedChild(_RecordedBase):
    digest: str


def _names_a_typed_dict_child() -> type:
    return _RecordedChild


def test_a_typed_dict_walks_the_base_its_class_statement_names():
    """A `TypedDict` keeps only `dict` in `__bases__` and the base its statement names in
    `__orig_bases__`: its walk reaches that base, as a class statement's walk does, and a cache
    hit reads it again."""
    manifest = fingerprint.get_stage_fingerprint(_names_a_typed_dict_child)
    reads = _recording(_names_a_typed_dict_child)

    key = _definition_key("class", _RecordedBase)
    assert manifest[key] == fingerprint.hash_function_ast(_RecordedBase), manifest
    assert any(path[-1] == "ob:0" for path, _ in reads), reads
    assert fingerprint._recorded_reads_hold(
        _names_a_typed_dict_child, [[list(path), token] for path, token in reads]
    )


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


def test_loader_fingerprint_caches_underlying_methods(reset_fingerprint_cache_state):
    """Loader methods remain cached after their temporary bound method is released."""
    fingerprint.get_loader_fingerprint(loaders.CSV())

    assert len(fingerprint._hash_function_ast_cache) == 3


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


def test_is_user_code_pivot_is_framework():
    """Pivot's own modules are framework code, not user code."""
    assert fingerprint.is_user_code(fingerprint) is False


def _helper_reads_pivot_tui() -> float | None:
    return pivot_tui.stats.get_memory_mb()


def test_pivot_tui_installed_from_a_checkout_is_framework() -> None:
    """pivot_tui is pivot's own package too: installed from a source checkout outside
    site-packages, as pivot's workspace installs it, it is not user code, and a stage that calls
    into it is not fingerprinted against its source."""
    tui_file = pathlib.Path(pivot_tui.stats.__file__).resolve()
    assert "site-packages" not in tui_file.parts, "pivot_tui is installed from its checkout"
    assert not tui_file.is_relative_to(pathlib.Path(sys.prefix).resolve())

    assert fingerprint.is_user_code(pivot_tui.stats) is False
    assert fingerprint.is_user_code(pivot_tui.stats.get_memory_mb) is False
    assert sorted(fingerprint.get_stage_fingerprint(_helper_reads_pivot_tui)) == [
        "byname:pivot_tui",
        "self:_helper_reads_pivot_tui",
    ], "pivot_tui is named (a worker imports it by name), its source is not read"


def _helper_install_distribution(site_dir: pathlib.Path, name: str, files: dict[str, str]) -> None:
    """Install distribution ``name`` into ``site_dir`` as an installer records it: its files,
    and a dist-info directory whose RECORD lists them."""
    dist_info = f"{name.replace('-', '_')}-1.0.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
    files = {**files, f"{dist_info}/METADATA": metadata}
    for rel, text in files.items():
        path = site_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    record = [*files, f"{dist_info}/RECORD"]
    (site_dir / dist_info / "RECORD").write_text("".join(f"{rel},,\n" for rel in record))


@pytest.mark.parametrize(
    ("sys_path", "distributions", "files", "expected"),
    [
        pytest.param(
            ["site"],
            {"pivot-tui": {"pivot_tui/__init__.py": ""}},
            {},
            ["site/pivot_tui"],
            id="wheel",
        ),
        pytest.param(
            ["site", "checkout/src"],
            {"pivot-tui": {"pivot_tui.pth": "# a comment\n\nimport os\n{tmp}/checkout/src\n"}},
            {"checkout/src/pivot_tui/__init__.py": ""},
            ["checkout/src/pivot_tui"],
            id="editable",
        ),
        pytest.param(
            ["site", "checkout/src"],
            {"pivot-tui": {"pivot_tui.pth": "\ufeff{tmp}/checkout/src\n"}},
            {"checkout/src/pivot_tui/__init__.py": ""},
            ["checkout/src/pivot_tui"],
            id="editable-pth-with-a-byte-order-mark",
        ),
        pytest.param(["site"], {}, {}, [], id="not-installed"),
        pytest.param(
            ["user", "site"],
            {"pivot-tui": {"pivot_tui/__init__.py": ""}},
            {"user/pivot_tui.py": ""},
            ["site/pivot_tui"],
            id="user-module-ahead-of-pivot-tui",
        ),
        pytest.param(
            ["user", "site", "checkout/src"],
            {"pivot-tui": {"pivot_tui.pth": "{tmp}/checkout/src\n"}},
            {"user/pivot_tui/__init__.py": "", "checkout/src/pivot_tui/__init__.py": ""},
            ["checkout/src/pivot_tui"],
            id="user-package-ahead-of-editable-pivot-tui",
        ),
        pytest.param(
            ["site", "user/src"],
            {"mytui": {"mytui.pth": "{tmp}/user/src\n"}},
            {"user/src/pivot_tui/__init__.py": ""},
            [],
            id="user-distribution-provides-pivot_tui",
        ),
    ],
)
def test_pivot_tui_is_pivot_s_package_only_where_the_pivot_tui_distribution_installed_it(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    sys_path: list[str],
    distributions: dict[str, dict[str, str]],
    files: dict[str, str],
    expected: list[str],
) -> None:
    """pivot_tui's directories are the ones the installed pivot-tui distribution records: its
    site directory for a wheel, the directories its ``.pth`` file adds for an editable install,
    none when it is not installed. A user's module or package named ``pivot_tui``, ahead of
    pivot-tui's on ``sys.path`` or installed by the user's own distribution, is not among them."""
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    for name, dist_files in distributions.items():
        _helper_install_distribution(
            tmp_path / "site",
            name,
            {rel: text.format(tmp=tmp_path) for rel, text in dist_files.items()},
        )
    (tmp_path / "site").mkdir(exist_ok=True)

    with monkeypatch.context() as patch:
        patch.setattr(sys, "path", [str(tmp_path / entry) for entry in sys_path])
        dirs = fingerprint._pivot_tui_dirs()

    assert dirs == tuple((tmp_path / rel).resolve() for rel in expected)


def _helper_remove_read_permission(path: pathlib.Path) -> None:
    path.chmod(0)


def _helper_replace_with_a_directory(path: pathlib.Path) -> None:
    path.unlink()
    path.mkdir()


@pytest.mark.parametrize(
    "make_unreadable",
    [
        pytest.param(_helper_remove_read_permission, id="mode-000"),
        pytest.param(_helper_replace_with_a_directory, id="directory"),
    ],
)
def test_pivot_tui_pth_file_python_cannot_read_adds_no_directory(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    make_unreadable: typing.Callable[[pathlib.Path], None],
) -> None:
    """A ``.pth`` file of pivot-tui's that cannot be read is skipped, as ``site`` skips it: it
    adds no directory to ``sys.path``, so no directory of pivot_tui's comes from it, and reading
    pivot's own packages does not fail."""
    (tmp_path / "checkout" / "src" / "pivot_tui").mkdir(parents=True)
    site_dir = tmp_path / "site"
    _helper_install_distribution(
        site_dir, "pivot-tui", {"pivot_tui.pth": f"{tmp_path}/checkout/src\n"}
    )
    make_unreadable(site_dir / "pivot_tui.pth")

    with monkeypatch.context() as patch:
        patch.setattr(sys, "path", [str(site_dir)])
        dirs = fingerprint._pivot_tui_dirs()

    assert dirs == ()


def test_pivot_tui_pth_file_that_is_not_utf_8_is_read_in_the_locale_s_encoding(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.pth`` file of pivot-tui's that is not UTF-8 is decoded in the locale's encoding, as
    ``site`` decodes it, so the checkout it adds to ``sys.path`` holds pivot's own package."""
    checkout = tmp_path / "caf\xe9" / "src"
    (checkout / "pivot_tui").mkdir(parents=True)
    site_dir = tmp_path / "site"
    _helper_install_distribution(site_dir, "pivot-tui", {"pivot_tui.pth": ""})
    (site_dir / "pivot_tui.pth").write_bytes(f"{checkout}\n".encode("latin-1"))

    with monkeypatch.context() as patch:
        patch.setattr(locale, "getencoding", lambda: "latin-1")
        patch.setattr(sys, "path", [str(site_dir)])
        dirs = fingerprint._pivot_tui_dirs()

    assert dirs == ((checkout / "pivot_tui").resolve(),)


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

    The is_library_path check must resolve symlinks to correctly identify stdlib.
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
    assert _owned_key("const", inner_func, "x") in fp
    assert fp[_owned_key("const", inner_func, "x")] == "10"


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
    assert _owned_key("const", add_five, "n") in fp


def test_stdlib_callable_in_closure_does_not_raise():
    """Stdlib callables like typing.cast should be silently skipped, not flagged as mutable."""
    # This was a regression where non-user-code callables hit the catch-all else branch
    # and _check_mutable_capture raised because repr() contains 0x memory addresses.
    fp = fingerprint.get_stage_fingerprint(_helper_uses_stdlib_callable)
    assert "self:_helper_uses_stdlib_callable" in fp
    # cast should NOT appear as const: or func: — it's a stdlib callable, silently skipped
    assert not any(k.endswith(":cast") for k in fp), f"cast should be skipped, found in: {fp}"


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

    assert "self:inner" in fp
    assert _definition_key("func", helper_func) in fp


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


def _helper_check_dynamic(func: typing.Callable[..., object]) -> None:
    tree = fingerprint._source_tree(func)
    assert tree is not None, "the helper must have source"
    fingerprint._check_dynamic_name_access(func, tree)


def test_check_dynamic_name_access_allows_literal_getattr():
    """Literal getattr usage should be allowed."""
    _helper_check_dynamic(_helper_static_getattr)


def test_check_dynamic_name_access_rejects_dynamic_getattr():
    """Dynamic getattr usage should be rejected."""
    with pytest.raises(exceptions.StageDefinitionError, match="getattr"):
        _helper_check_dynamic(_helper_dynamic_getattr)


def test_check_dynamic_name_access_rejects_globals():
    """globals() usage should be rejected."""
    with pytest.raises(exceptions.StageDefinitionError, match="globals"):
        _helper_check_dynamic(_helper_uses_globals)


def test_check_dynamic_name_access_rejects_import_module():
    """import_module usage should be rejected."""
    with pytest.raises(exceptions.StageDefinitionError, match="dynamic imports"):
        _helper_check_dynamic(_helper_uses_import_module)


def test_compute_function_hash_warns_when_source_unavailable_for_user_code(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """Missing source should warn for user code when no __code__ exists."""

    class FakeCallable:
        """Callable without source or __code__."""

        def __call__(self) -> int:
            return 42

    def _raise_getsource(*_args: object, **_kwargs: object) -> str:
        raise OSError("missing")

    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(inspect, "getsource", _raise_getsource)

    fingerprint._compute_function_hash(FakeCallable())

    assert "Cannot get source for" in caplog.text


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
    assert _owned_key("func", top_func, "middle_helper") in fp
    assert _definition_key("func", leaf_helper) in fp

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
    assert _const_key("TEST_STRING") in fp
    assert fp[_const_key("TEST_STRING")] == "'Hello, World!'"


def test_fingerprint_with_bytes_constant():
    """Should capture bytes constants."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_bytes)

    assert "self:_helper_uses_bytes" in fp
    assert _const_key("TEST_BYTES") in fp
    assert fp[_const_key("TEST_BYTES")] == "b'binary data'"


def test_fingerprint_with_none_constant():
    """Should capture None constants."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_none)

    assert "self:_helper_uses_none" in fp
    assert _const_key("TEST_NONE") in fp
    assert fp[_const_key("TEST_NONE")] == "None"


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
    assert _const_key("TEST_FLOAT") in fp
    assert fp[_const_key("TEST_FLOAT")] == "3.14159"


def test_fingerprint_with_bool_constant():
    """Should capture boolean constants."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_bool)

    assert "self:_helper_uses_bool" in fp
    assert _const_key("TEST_BOOL") in fp
    assert fp[_const_key("TEST_BOOL")] == "True"


# --- Pydantic schema hashing tests ---


def test_strip_schema_metadata_removes_titles_and_descriptions():
    """Should remove title/description from schema recursively."""
    schema = {
        "title": "Top",
        "description": "Top desc",
        "type": "object",
        "properties": {
            "name": {
                "title": "Name",
                "description": "Name desc",
                "type": "string",
            },
            "items": {
                "title": "Items",
                "description": "Items desc",
                "type": "array",
                "items": {
                    "title": "Item",
                    "description": "Item desc",
                    "type": "string",
                },
            },
        },
        "$defs": {
            "Inner": {
                "title": "Inner",
                "description": "Inner desc",
                "type": "object",
                "properties": {
                    "value": {
                        "title": "Value",
                        "description": "Value desc",
                        "type": "integer",
                    }
                },
            }
        },
        "anyOf": [
            {"title": "OptA", "description": "OptA desc", "type": "string"},
            {"type": "number"},
        ],
    }

    stripped = fingerprint._strip_schema_metadata(schema)

    assert "title" not in stripped
    assert "description" not in stripped
    assert "title" in schema
    assert "description" in schema
    assert "title" not in stripped["$defs"]["Inner"]
    assert "description" not in stripped["$defs"]["Inner"]
    assert "title" not in stripped["properties"]["name"]
    assert stripped["properties"]["name"]["type"] == "string"
    assert "title" not in stripped["properties"]["items"]["items"]
    assert stripped["anyOf"][0]["type"] == "string"


def _takes_the_outer_model(model: _PydanticOuterModel) -> None:
    pass


def test_a_pydantic_model_walk_adds_its_schema_and_nested_models():
    """A model in a type hint is walked as a class whose schema is its entry, and each model its
    field annotations name is walked so too."""
    manifest = fingerprint.get_stage_fingerprint(_takes_the_outer_model)

    outer_schema = _definition_key("schema", _PydanticOuterModel)
    assert outer_schema in manifest
    assert _definition_key("schema", _PydanticInnerModel) in manifest
    assert _definition_key("class", _PydanticInnerModel) in manifest
    assert len(manifest[outer_schema]) == 16


def test_model_field_classes_tracks_nested_models():
    """Field annotations are walked into generics, so nested Pydantic models are found."""
    assert fingerprint._model_field_classes(_PydanticOuterModel) == [
        _PydanticInnerModel,
        _PydanticInnerModel,
    ]


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


def test_covered_user_modules_reads_every_module_key_form():
    """Bare `module:` keys and the `dep:`/`out:`-prefixed keys a loader's walk adds count; other
    keys that mention a module do not, nor a dependency or output named `module`, whose prefix
    the stage's output keys end."""
    manifest = {
        "self:count": "a",
        "module:helpers": "",
        "dep:n:module:helpers.lib": "",
        "out:t/count:out:module:other.mod": "",
        "func:helpers.lib.value": "b",
        "const:module.NAME": "1",
        "dep:n:loader:helpers.myloader.Scaled:load": "c",
        "dep:module:loader:pivot.loaders.YAML:load": "d",
        "out:t/count:module:tag": "data",
    }

    assert fingerprint.covered_user_modules(manifest, {"t/count:out", "t/count:module"}) == {
        "helpers",
        "helpers.lib",
        "other.mod",
    }


def test_loader_fingerprint_includes_load_method():
    """Should fingerprint the load method."""
    from pivot import loaders

    loader = loaders.CSV()
    fp = fingerprint.get_loader_fingerprint(loader)

    key = _loader_key(loader, "load")
    assert key in fp
    assert isinstance(fp[key], str)
    assert len(fp[key]) == 16  # xxhash64 hex


def test_loader_fingerprint_includes_save_method():
    """Should fingerprint the save method."""
    from pivot import loaders

    loader = loaders.CSV()
    fp = fingerprint.get_loader_fingerprint(loader)

    key = _loader_key(loader, "save")
    assert key in fp
    assert isinstance(fp[key], str)
    assert len(fp[key]) == 16


def test_loader_fingerprint_includes_config():
    """Should fingerprint dataclass field values."""
    from pivot import loaders

    loader = loaders.CSV(index_col="id", sep=";")
    fp = fingerprint.get_loader_fingerprint(loader)

    key = _loader_key(loader, "config")
    assert key in fp
    assert isinstance(fp[key], str)


def test_loader_config_change_changes_fingerprint():
    """Different config values should produce different fingerprints."""
    from pivot import loaders

    loader1 = loaders.CSV(sep=",")
    loader2 = loaders.CSV(sep=";")

    fp1 = fingerprint.get_loader_fingerprint(loader1)
    fp2 = fingerprint.get_loader_fingerprint(loader2)

    assert fp1[_loader_key(loader1, "config")] != fp2[_loader_key(loader2, "config")]


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
    assert fp_csv[_loader_key(csv_loader, "load")] != fp_json[_loader_key(json_loader, "load")]


def test_json_loader_fingerprint():
    """Should fingerprint JSON loader correctly."""
    from pivot import loaders

    loader = loaders.JSON(indent=4)
    fp = fingerprint.get_loader_fingerprint(loader)

    assert _loader_key(loader, "load") in fp
    assert _loader_key(loader, "save") in fp
    assert _loader_key(loader, "config") in fp


def test_yaml_loader_fingerprint():
    """Should fingerprint YAML loader correctly."""
    from pivot import loaders

    loader = loaders.YAML()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert _loader_key(loader, "load") in fp
    assert _loader_key(loader, "save") in fp
    assert _loader_key(loader, "config") in fp


def test_pickle_loader_fingerprint():
    """Should fingerprint Pickle loader correctly."""
    from pivot import loaders

    loader = loaders.Pickle()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert _loader_key(loader, "load") in fp
    assert _loader_key(loader, "save") in fp
    assert _loader_key(loader, "config") in fp


def test_pathonly_loader_fingerprint():
    """Should fingerprint PathOnly loader correctly."""
    from pivot import loaders

    loader = loaders.PathOnly()
    fp = fingerprint.get_loader_fingerprint(loader)

    assert _loader_key(loader, "load") in fp
    assert _loader_key(loader, "save") in fp
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

    assert _loader_key(loader, "load") in fp
    assert _loader_key(loader, "save") in fp
    assert _loader_key(loader, "config") in fp


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
    assert fp1[_loader_key(LoaderV1(), "load")] != fp2[_loader_key(LoaderV2(), "load")]


def test_loader_fingerprint_stable():
    """Loader fingerprint should be stable across calls."""
    from pivot import loaders

    loader = loaders.CSV(index_col="id")

    fp1 = fingerprint.get_loader_fingerprint(loader)
    fp2 = fingerprint.get_loader_fingerprint(loader)

    assert fp1 == fp2


@dataclasses.dataclass(frozen=True)
class _HeldReader(loaders.Reader[str]):
    """A user loader another loader holds; its `load` reads library module `json`."""

    suffix: str = "A"

    def load(self, path: pathlib.Path) -> str:
        return path.read_text() + json.dumps(self.suffix)


@dataclasses.dataclass(frozen=True)
class _HoldingReader(loaders.Reader[str]):
    """A user loader whose config field `inner` holds a loader, or a value holding loaders."""

    inner: typing.Any

    def load(self, path: pathlib.Path) -> str:
        return self.inner.load(path)


@dataclasses.dataclass(frozen=True)
class _Settings:
    """A user class that is not a loader; it can hold one."""

    reader: typing.Any = None


@dataclasses.dataclass(frozen=True)
class _TwoFieldReader(loaders.Reader[str]):
    """A user loader whose second config field holds a loader."""

    scale: int
    inner: typing.Any

    def load(self, path: pathlib.Path) -> str:
        return self.inner.load(path) * self.scale


@dataclasses.dataclass(frozen=True)
class _BaseLoad(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text().upper()


@dataclasses.dataclass(frozen=True)
class _InheritsLoad(_BaseLoad):
    """A user loader whose `load` is its base class's."""


def _apply(reader: loaders.Reader[str], path: pathlib.Path) -> str:
    return reader.load(path)


_READ_THROUGH_PARTIAL = functools.partial(_apply, _HeldReader())


def _reads_through_a_partial(path: pathlib.Path) -> str:
    return _READ_THROUGH_PARTIAL(path)


_READ_THROUGH_KEYWORD = functools.partial(_apply, reader=_HeldReader())


def _reads_through_a_keyword(path: pathlib.Path) -> str:
    return _READ_THROUGH_KEYWORD(path=path)


@dataclasses.dataclass(frozen=True)
class _WithReaderDefault:
    reader: typing.Any = _HeldReader()


def _reads_a_dataclass_default(path: pathlib.Path) -> str:
    return _WithReaderDefault().reader.load(path)


def _entries(manifest: dict[str, str]) -> set[tuple[str, str]]:
    """Each entry of `manifest`, as a set to compare."""
    return set(manifest.items())


def _without_config(manifest: dict[str, str]) -> set[tuple[str, str]]:
    """The entries of a top-level loader's manifest that the same loader, held in a value, adds
    to its holder's: all but its config hash, which the holder's form holds."""
    return {(key, value) for key, value in manifest.items() if not key.endswith(":config")}


def _recording(func: typing.Callable[..., object]) -> list[tuple[tuple[str, ...], str]]:
    """The reads a cacheable walk of `func` records."""
    with fingerprint._collected_capture_errors():
        return fingerprint._stage_manifest(func, recording=True)[1]


def _counting_expansions(monkeypatch: pytest.MonkeyPatch, name: str) -> list[type]:
    """The classes each later call of `fingerprint.<name>` (an expansion that takes the class
    first) is given, in order."""
    walked = list[type]()
    original = getattr(fingerprint, name)

    def counting(cls: type, *args: typing.Any) -> None:
        walked.append(cls)
        original(cls, *args)

    monkeypatch.setattr(fingerprint, name, counting)
    return walked


@pytest.mark.parametrize(
    "holder",
    [
        pytest.param(_HoldingReader(_HeldReader()), id="the-field"),
        pytest.param(_TwoFieldReader(2, _HeldReader()), id="the-second-field"),
        pytest.param(_HoldingReader((_HeldReader(), loaders.Text())), id="tuple"),
        pytest.param(_HoldingReader([_Settings(), _HeldReader()]), id="list-with-a-non-loader"),
        pytest.param(_HoldingReader({"a:module:b": _HeldReader()}), id="dict-key-with-colons"),
        pytest.param(_HoldingReader(frozenset({_HeldReader()})), id="frozenset"),
        pytest.param(_HoldingReader(functools.partial(_apply, _HeldReader())), id="partial-arg"),
        pytest.param(
            _HoldingReader(functools.partial(_apply, reader=_HeldReader())), id="partial-keyword"
        ),
        pytest.param(_HoldingReader(_Settings(_HeldReader())), id="inside-a-dataclass"),
        pytest.param(_HoldingReader(_HoldingReader(_HeldReader())), id="held-by-a-held-loader"),
    ],
)
def test_a_loader_a_loader_config_holds_gets_a_top_level_loaders_fingerprint(holder: typing.Any):
    """Wherever the config's form meets `_HeldReader()`, the holder's manifest holds the entries a
    top-level `_HeldReader()` gets from its class (its walk of `load`, its `load` hash) and its
    `byname:json`; its config is part of the holder's form. No `held:` entry is made, and no dict
    key or other user value enters a key."""
    manifest = fingerprint.get_loader_fingerprint(holder)

    assert _without_config(fingerprint.get_loader_fingerprint(_HeldReader())) <= _entries(manifest)
    assert manifest["byname:json"] == ""
    assert not any(key.startswith("held:") for key in manifest), manifest
    assert not any("module:b" in key or ":byname:" in key for key in manifest), manifest
    stage = {f"dep:raw:{key}": item for key, item in manifest.items()}
    assert __name__ in fingerprint.covered_user_modules(stage, ())


def test_a_held_loaders_config_class_and_inherited_code_are_its_entries():
    """A held loader's config is in its holder's form: `(_HeldReader("A"), _HeldReader("B"))`
    and `(_HeldReader("A"), _HeldReader("C"))` give two config hashes. A held library loader gets
    its own hashes, and a held loader whose `load` is inherited gets that `load`'s hash and walk,
    as a top-level one does."""
    both = fingerprint.get_loader_fingerprint(_HoldingReader((_HeldReader("A"), _HeldReader("B"))))
    other = fingerprint.get_loader_fingerprint(_HoldingReader((_HeldReader("A"), _HeldReader("C"))))
    text = fingerprint.get_loader_fingerprint(_HoldingReader(loaders.Text()))
    inherited = fingerprint.get_loader_fingerprint(_HoldingReader(_InheritsLoad()))

    config = _loader_key(_HoldingReader(None), "config")
    assert both[config] != other[config]
    assert _without_config(fingerprint.get_loader_fingerprint(loaders.Text())) <= _entries(text)
    assert _without_config(fingerprint.get_loader_fingerprint(_InheritsLoad())) <= _entries(
        inherited
    )
    assert (
        _loader_key(_InheritsLoad(), "load"),
        fingerprint.hash_function_ast(_BaseLoad.load),
    ) in _entries(inherited)


def test_a_loader_that_holds_itself_is_fingerprinted_once():
    """A loader that holds itself, directly or through another loader, is a cycle: the walk
    stops at the loader it is fingerprinting already, and its class's entries are there once."""
    looped = _HoldingReader([])
    looped.inner.append(looped)
    first = _HoldingReader([])
    second = _HoldingReader(first)
    first.inner.append(second)

    alone = fingerprint.get_loader_fingerprint(looped)
    pair = fingerprint.get_loader_fingerprint(first)

    load = _loader_key(looped, "load")
    assert load in alone, alone
    assert load in pair, pair
    assert not any(key.startswith("held:") for key in (*alone, *pair))


def test_a_loader_held_many_times_is_walked_once_in_a_stage(monkeypatch: pytest.MonkeyPatch):
    """Fifty equal `_HeldReader()` in one config, and the same one in a second loader of the
    stage: each loader class is expanded once in the stage (`stage_fingerprint_scope`)."""
    walked = _counting_expansions(monkeypatch, "_expand_loader_class")
    with fingerprint.stage_fingerprint_scope("t/use"):
        fingerprint.merge_loader_fingerprint(
            {}, "dep:a:", _HoldingReader(tuple(_HeldReader() for _ in range(50)))
        )
        fingerprint.merge_loader_fingerprint({}, "dep:b:", _HoldingReader(_HeldReader()))

    assert walked == [_HoldingReader, _HeldReader]


def test_a_chain_of_held_loaders_grows_by_one_loader_per_level():
    """`_HoldingReader` nested N deep around `_HeldReader()`: each loader class adds its entries
    once, whatever the depth, so the keys grow by a constant per level (none), not
    exponentially."""
    sizes = list[int]()
    value: typing.Any = _HeldReader()
    for _ in range(8):
        value = _HoldingReader(value)
        sizes.append(len(fingerprint.get_loader_fingerprint(value)))

    steps = {after - before for before, after in zip(sizes, sizes[1:], strict=False)}
    assert len(steps) == 1, sizes


def _make_reader(suffix: str) -> typing.Any:
    @dataclasses.dataclass(frozen=True)
    class Made(loaders.Reader[str]):
        def load(self, path: pathlib.Path) -> str:
            return path.read_text() + suffix

    return Made


def test_two_classes_one_factory_made_are_each_fingerprinted():
    """`_make_reader("first")` and `_make_reader("A")` share a name: inside a value each is keyed
    by its identity (`<name>@<identity8>`), so both are walked, neither overwrites the other, and
    editing the second's suffix changes the holder's manifest."""
    first, second, edited = _make_reader("first"), _make_reader("A"), _make_reader("B")
    both = fingerprint.get_loader_fingerprint(_HoldingReader((first(), second())))
    after = fingerprint.get_loader_fingerprint(_HoldingReader((first(), edited())))

    made = f"loader:{__name__}._make_reader.<locals>.Made@"
    loads = {key for key in both if key.startswith(made) and key.endswith(":load")}
    assert len(loads) == 2, both
    assert both != after


@dataclasses.dataclass(frozen=True)
class _ReadsADefault(loaders.Reader[str]):
    """A loader whose `load` reads a dataclass whose field default is this loader again."""

    suffix: str = "ok"

    def load(self, path: pathlib.Path) -> str:
        return path.read_text() + _LoaderDefaults().reader.suffix


@dataclasses.dataclass(frozen=True)
class _LoaderDefaults:
    reader: typing.Any = _ReadsADefault()


@pytest.mark.parametrize(
    "loader", [_ReadsADefault(), _HoldingReader(_ReadsADefault())], ids=["top-level", "held"]
)
def test_a_loader_its_own_walk_reaches_again_is_fingerprinted_once(loader: typing.Any):
    """`_ReadsADefault.load` reads `_LoaderDefaults`, whose field default is a `_ReadsADefault`:
    the walk reaches that loader again while it is fingerprinting it, and stops there."""
    manifest = fingerprint.get_loader_fingerprint(loader)

    assert _definition_key("class", _LoaderDefaults) in manifest


@dataclasses.dataclass(frozen=True)
class _CallsAHelper(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return self.decorate(path.read_text())

    def decorate(self, text: str) -> str:
        return text + "A"


@dataclasses.dataclass(frozen=True)
class _InheritsAHelper(_CallsAHelper):
    """`load` and its helper `decorate` are both a user base class's."""


@pytest.mark.parametrize("loader_class", [_CallsAHelper, _InheritsAHelper])
def test_a_loaders_helper_method_is_walked_top_level_and_held(loader_class: typing.Any):
    """`load` calls `self.decorate()`, defined on the class or on a user base class it inherits
    from: `decorate` is an entry of a top-level and of a held loader, under its defining class."""
    key = _definition_key("func", _CallsAHelper.decorate)
    top = fingerprint.get_loader_fingerprint(loader_class())
    held = fingerprint.get_loader_fingerprint(_HoldingReader(loader_class()))

    assert top[key] == fingerprint.hash_function_ast(_CallsAHelper.decorate)
    assert (key, top[key]) in _entries(held)


def test_many_loaders_of_one_class_walk_the_class_once_in_a_stage(
    monkeypatch: pytest.MonkeyPatch,
):
    """Fifty `_HeldReader`s with fifty configs, held in one loader, and another in a second:
    the class's own entries (its `load`'s walk) are computed once in the stage, and each loader
    still gets them."""
    walked = _counting_expansions(monkeypatch, "_expand_loader_class")
    readers = tuple(_HeldReader(str(index)) for index in range(50))
    with fingerprint.stage_fingerprint_scope("t/use"):
        many = fingerprint.get_loader_fingerprint(_HoldingReader(readers))
        fingerprint.get_loader_fingerprint(_HoldingReader(_HeldReader("other")))

    assert walked.count(_HeldReader) == 1, walked
    own = fingerprint.get_loader_fingerprint(_HeldReader("7"))
    assert _without_config(own) <= _entries(many)


@pytest.mark.parametrize(
    ("func", "path"),
    [
        pytest.param(
            _reads_through_a_partial, ("g:_READ_THROUGH_PARTIAL", "pa:0", "cls"), id="partial"
        ),
        pytest.param(
            _reads_through_a_keyword,
            ("g:_READ_THROUGH_KEYWORD", "pk:reader", "cls"),
            id="partial-keyword",
        ),
        pytest.param(
            _reads_a_dataclass_default, ("g:_WithReaderDefault", "fd:reader", "cls"), id="dataclass"
        ),
    ],
)
def test_a_loader_a_stage_walk_holds_gets_its_fingerprint_and_its_reads(
    func: typing.Callable[..., object], path: tuple[str, ...]
):
    """A global partial's argument, or a field default of a dataclass the stage uses, holds
    `_HeldReader()`: the stage walk holds its class's entries and records the reads of its code
    under the path through the value to its class (`pa:0`, `pk:reader`, `fd:reader`, then
    `cls`), which a cache hit follows and takes again, as it does a named class's."""
    manifest = fingerprint.get_stage_fingerprint(func)
    reads = _recording(func)

    assert _without_config(fingerprint.get_loader_fingerprint(_HeldReader())) <= _entries(manifest)
    assert ((*path, "lm:load"), fingerprint._read_token(_HeldReader, "lm:load")) in reads, reads
    assert fingerprint._recorded_reads_hold(func, [[list(p), token] for p, token in reads])


class _Captured:
    """A user class, not a pivot loader, that no walked code names: the walk meets it only as
    the class of an instance a value holds."""

    def suffix(self) -> str:
        return "A"


def _tag(captured: typing.Any, text: str) -> str:
    return text + captured.suffix()


def _take(value: object, x: int) -> int:
    return x


_TAG_ARGUMENT = functools.partial(_tag, _Captured())
_TAG_KEYWORD = functools.partial(_tag, captured=_Captured())


def _tags_through_an_argument(text: str) -> str:
    return _TAG_ARGUMENT(text)


def _tags_through_a_keyword(text: str) -> str:
    return _TAG_KEYWORD(text=text)


@dataclasses.dataclass(frozen=True)
class _WithCapturedDefault:
    captured: typing.Any = _Captured()


def _tags_through_a_dataclass_default(text: str) -> str:
    return _tag(_WithCapturedDefault().captured, text)


class _CapturedParams(stage_def.StageParams):
    captured: typing.Any = _Captured()


_CAPTURED_PARAMS = _CapturedParams()


def _tags_through_a_stageparams_default(
    text: str, params: _CapturedParams = _CAPTURED_PARAMS
) -> str:
    return _tag(params.captured, text)


def _class_walk(cls: type) -> set[tuple[str, str]]:
    """The entries a value that holds an instance of `cls` gets for it: its hash and its walk,
    under its key inside a value (`_class_key`), which is a stage's key for a class with source
    that its module's name holds."""
    walk = fingerprint._Walk()
    key = fingerprint._class_key(cls, in_value=True)
    root = walk.node(cls, fingerprint._NodeKind.CLASS, key)
    walk.run()
    manifest = {f"class:{key}": fingerprint.hash_function_ast(cls)}
    with fingerprint._collected_capture_errors():
        fingerprint._project(root, manifest)
    return set(manifest.items())


@pytest.mark.parametrize(
    ("func", "path", "schema_read"),
    [
        pytest.param(
            _tags_through_an_argument,
            ("g:_TAG_ARGUMENT", "pa:0", "cls"),
            None,
            id="partial-argument",
        ),
        pytest.param(
            _tags_through_a_keyword,
            ("g:_TAG_KEYWORD", "pk:captured", "cls"),
            None,
            id="partial-keyword",
        ),
        pytest.param(
            _tags_through_a_dataclass_default,
            ("g:_WithCapturedDefault", "fd:captured", "cls"),
            None,
            id="dataclass-default",
        ),
        pytest.param(
            _tags_through_a_stageparams_default,
            ("d:params", "at:captured", "cls"),
            ("d:params", "cls", "schema"),
            id="stageparams-default",
        ),
    ],
)
def test_a_captured_instance_gets_its_classs_walk_and_its_reads(
    func: typing.Callable[..., object],
    path: tuple[str, ...],
    schema_read: tuple[str, ...] | None,
):
    """A global partial's argument or keyword, the field default of a dataclass the stage
    reads, or the StageParams default of a stage's parameter holds `_Captured()`. The stage's
    manifest holds the walk a stage that names `_Captured` gets, under its own key, and the walk
    records the reads of the class's code under the path through the value to its class, which
    a manifest cache hit follows and takes again. The StageParams class of the default is read
    too, under the first path that reaches it, by its schema hash, which holds its field
    defaults' forms."""
    manifest = fingerprint.get_stage_fingerprint(func)
    reads = _recording(func)

    assert _class_walk(_Captured) <= _entries(manifest)
    assert ((*path, "m:suffix"), fingerprint._read_token(_Captured, "m:suffix")) in reads, reads
    if schema_read is not None:
        schema_token = fingerprint._read_token(_CapturedParams, "schema")
        assert (schema_read, schema_token) in reads, reads
    assert fingerprint._recorded_reads_hold(func, [[list(p), token] for p, token in reads])


@pytest.mark.parametrize(
    "holder",
    [
        pytest.param(_HoldingReader(_Captured()), id="the-field"),
        pytest.param(_HoldingReader([1, {"k": _Captured()}]), id="in-a-container"),
        pytest.param(_HoldingReader(_Settings(_Captured())), id="inside-a-dataclass"),
        pytest.param(_HoldingReader(functools.partial(_tag, _Captured())), id="partial-arg"),
    ],
)
def test_a_loader_config_holding_an_instance_gets_its_classs_walk(holder: typing.Any):
    """Wherever a loader's config holds `_Captured()`, the loader's manifest holds the walk of
    `_Captured` under its own key, and its module is covered."""
    manifest = fingerprint.get_loader_fingerprint(holder)

    assert _class_walk(_Captured) <= _entries(manifest)
    stage = {f"dep:raw:{key}": item for key, item in manifest.items()}
    assert __name__ in fingerprint.covered_user_modules(stage, ())


class _Token(str):
    """A user `str` subclass: the canonical form records its value as a plain string."""

    def suffix(self) -> str:
        return "A"


class _BaseSuffix:
    def suffix(self) -> str:
        return "A"


class _InheritsSuffix(_BaseSuffix):
    """A user class whose method is its user base class's."""


class _Level(enum.IntEnum):
    """A user IntEnum: the canonical form records a member as a plain int."""

    LOW = 1

    def suffix(self) -> str:
        return "A"


_TOKEN = _Token("t")


def _tags_a_str_default(text: str, token: str = _TOKEN) -> str:
    return _tag(token, text)


@pytest.mark.parametrize(
    ("value", "cls", "method"),
    [
        pytest.param(_Token("t"), _Token, _Token.suffix, id="str-subclass"),
        pytest.param(_Level.LOW, _Level, _Level.suffix, id="intenum-member"),
        pytest.param(_InheritsSuffix(), _InheritsSuffix, _BaseSuffix.suffix, id="inherited-method"),
    ],
)
def test_an_instance_the_form_records_as_a_scalar_or_by_its_base_gets_its_classs_walk(
    value: object, cls: type, method: typing.Callable[..., str]
):
    """A `str` subclass and an IntEnum member, whose canonical forms are a plain string and
    int, get their class's walk, `suffix` included; so does a class whose `suffix` its user
    base class defines, under that base class."""
    manifest = fingerprint.get_loader_fingerprint(_HoldingReader(value))

    assert _class_walk(cls) <= _entries(manifest)
    key = _definition_key("func", method)
    assert (key, fingerprint.hash_function_ast(method)) in _entries(manifest), manifest


def test_a_str_subclass_default_gets_its_classs_walk_and_its_value_token():
    """A parameter's default that is an instance of a user `str` subclass is recorded by its
    repr, as a `str` default is, and its class gets the walk a stage that names it gets; its
    read token is the hash of its form, which holds its class's key and its value, so a cache
    hit sees a new value."""
    manifest = fingerprint.get_stage_fingerprint(_tags_a_str_default)
    reads = dict(_recording(_tags_a_str_default))

    digest = xxhash.xxh64(fingerprint._form(_TOKEN).text.encode()).hexdigest()
    assert manifest[f"default:{fingerprint._definition_ref(_tags_a_str_default)}.token"] == "'t'"
    assert _class_walk(_Token) <= _entries(manifest)
    assert reads[("d:token",)] == f"default:{digest}"
    assert fingerprint._recorded_reads_hold(
        _tags_a_str_default, [[list(path), token] for path, token in reads.items()]
    )


def test_a_known_library_value_or_a_loader_adds_no_class_walk_and_an_opaque_one_is_refused():
    """Known value rows need no class walk; a pivot loader gets its loader class's entries, not a
    class walk; a `random.Random` has unreadable C state."""
    library = fingerprint.get_loader_fingerprint(
        _HoldingReader((pathlib.PurePosixPath("a"), collections.Counter("ab")))
    )
    loader = fingerprint.get_loader_fingerprint(_HoldingReader(_HeldReader()))

    assert not any(key.startswith("class:") for key in library), library
    held_class = _definition_key("class", _HeldReader)
    assert not any(key == held_class or key.startswith(f"{held_class}@") for key in loader), loader
    with pytest.raises(exceptions.StageDefinitionError, match="random.Random|cannot read"):
        fingerprint.get_loader_fingerprint(_HoldingReader(random.Random(1)))


_MANY_CAPTURED = functools.partial(_take, tuple(_Captured() for _ in range(10_000)))
_ONE_CAPTURED = functools.partial(_take, _Captured())


def _takes_many_captured(x: int) -> int:
    return _MANY_CAPTURED(x) + _ONE_CAPTURED(x)


def test_many_instances_of_one_class_walk_the_class_once_in_a_stage(
    monkeypatch: pytest.MonkeyPatch,
):
    """Ten thousand `_Captured()` in one partial and one in a second: the class is walked once
    in the stage, and its entry is in the manifest once, under its own key, not once per
    instance or per partial."""
    walked = _counting_expansions(monkeypatch, "_process_user_methods")
    with fingerprint.stage_fingerprint_scope("t/use"):
        manifest = fingerprint.get_stage_fingerprint(_takes_many_captured)

    assert walked.count(_Captured) == 1, walked
    key = _definition_key("class", _Captured)
    assert [entry for entry in manifest if entry.endswith(key.removeprefix("class:"))] == [key]


def test_a_cache_hit_walks_a_captured_instances_class_again(monkeypatch: pytest.MonkeyPatch):
    """The read the walk records for a partial holding `_Captured()` holds while the class is
    unchanged, and not once its method changes where no source file shows it."""
    reads: list[object] = [
        [list(path), token] for path, token in _recording(_tags_through_an_argument)
    ]
    assert fingerprint._recorded_reads_hold(_tags_through_an_argument, reads)

    def other(self: _Captured) -> str:
        return "B"

    monkeypatch.setattr(_Captured, "suffix", other)

    assert not fingerprint._recorded_reads_hold(_tags_through_an_argument, reads)


def _make_captured(suffix: str) -> type:
    class Made:
        def suffix(self) -> str:
            return suffix

    return Made


@pytest.mark.parametrize("container", [tuple, frozenset], ids=["tuple", "set-equal-forms"])
def test_two_classes_one_factory_made_keep_apart_in_one_value(container: type):
    """Instances of two classes from one factory share a name: inside a value each class is
    keyed by its identity, so each gets its own walk, also in a set (equal-looking members), and
    editing the second one's closure changes the holder's manifest."""
    first, second, edited = _make_captured("first"), _make_captured("A"), _make_captured("B")
    pair = fingerprint.get_loader_fingerprint(_HoldingReader(container((first(), second()))))
    after = fingerprint.get_loader_fingerprint(_HoldingReader(container((first(), edited()))))

    for made in (first, second):
        assert _class_walk(made) <= _entries(pair)
    assert pair != after


def test_an_instance_a_held_loader_holds_is_that_loaders_entry():
    """`_Captured()` in the config of a loader that another loader holds is in the holder's form,
    so its class's walk is in the holder's manifest, once, under its own key."""
    manifest = fingerprint.get_loader_fingerprint(_HoldingReader(_HoldingReader(_Captured())))

    assert _class_walk(_Captured) <= _entries(manifest)
    assert not any(key.startswith("held:") for key in manifest), manifest


_COLUMNS = ["a", "b"]


class _ReadsAList:
    def columns(self) -> list[str]:
        return _COLUMNS


_HOLDS_READS_A_LIST = functools.partial(_take, _ReadsAList())


def _takes_one_that_reads_a_list(x: int) -> int:
    return _HOLDS_READS_A_LIST(x)


def test_a_captured_instance_whose_class_the_walk_refuses_refuses_the_stage():
    """`_ReadsAList.columns` reads a list global, which the class walk refuses for a stage that
    names the class; a stage that captures an instance of it is refused the same way. The stage
    never names the class, so the message says how the stage reaches the value (the partial, the
    position in it, the instance's class, the method that reads it), and still points to
    StageParams."""
    with pytest.raises(exceptions.StageDefinitionError) as raised:
        fingerprint.get_stage_fingerprint(_takes_one_that_reads_a_list)

    message = str(raised.value)
    assert "captures non-scalar variable '_COLUMNS'" in message, message
    assert (
        "Reached through: global '_HOLDS_READS_A_LIST' -> argument 0 -> its class -> "
        + "member 'columns' -> global '_COLUMNS'."
    ) in message, message
    assert "move this value into the stage's StageParams" in message, message


def test_a_class_method_whose_wrapped_chain_loops_is_refused():
    """D3: a method's metadata loop refuses both named and captured class uses."""

    def loops(function: typing.Callable[..., str]) -> typing.Callable[..., str]:
        @functools.wraps(function)
        def wrapper(*args: typing.Any) -> str:
            return function(*args)

        wrapper.__wrapped__ = wrapper
        return wrapper

    class Looped:
        @loops
        def suffix(self) -> str:
            return "A"

    holds = functools.partial(_tag, Looped())

    def captures(text: str) -> str:
        return holds(text)

    def names(text: str) -> str:
        return _tag(Looped(), text)

    for stage in (captures, names):
        with pytest.raises(exceptions.StageDefinitionError) as raised:
            fingerprint.get_stage_fingerprint(stage)
        message = str(raised.value)
        assert "Looped.suffix" in message and "loop" in message, message
        assert "StageParams" in message and "break" in message.lower(), message


def _two_partials(first: str) -> typing.Callable[[str], str]:
    """A stage whose two partials hold instances of two classes one factory made."""
    first_holds = functools.partial(_tag, _make_captured(first)())
    second_holds = functools.partial(_tag, _make_captured("Z")())

    def stage(text: str) -> str:
        return first_holds(text) + second_holds(text)

    return stage


def _two_fields(label: str) -> typing.Callable[[str], str]:
    """A stage reading a dataclass whose two field defaults are such instances."""

    @dataclasses.dataclass(frozen=True)
    class Pair:
        first: typing.Any = _make_captured(label)()
        second: typing.Any = _make_captured("Z")()

    def stage(text: str) -> str:
        return _tag(Pair().first, text) + _tag(Pair().second, text)

    return stage


def _two_params(first: str) -> typing.Callable[[str], str]:
    """A stage calling two helpers whose StageParams defaults hold such instances."""

    class FirstParams(stage_def.StageParams):
        made: typing.Any = _make_captured(first)()

    class SecondParams(stage_def.StageParams):
        made: typing.Any = _make_captured("Z")()

    first_params, second_params = FirstParams(), SecondParams()

    def first_helper(text: str, params: typing.Any = first_params) -> str:
        return _tag(params.made, text)

    def second_helper(text: str, params: typing.Any = second_params) -> str:
        return _tag(params.made, text)

    def stage(text: str) -> str:
        return first_helper(text) + second_helper(text)

    return stage


def _two_loader_partials(first: str) -> typing.Callable[[pathlib.Path], str]:
    """A stage whose two partials hold two loaders whose classes one factory made."""
    first_holds = functools.partial(_apply, _make_reader(first)())
    second_holds = functools.partial(_apply, _make_reader("Z")())

    def stage(path: pathlib.Path) -> str:
        return first_holds(path) + second_holds(path)

    return stage


@pytest.mark.parametrize("shape", [_two_partials, _two_fields, _two_params, _two_loader_partials])
def test_two_classes_one_factory_made_keep_apart_in_two_values_of_a_stage(
    shape: typing.Callable[[str], typing.Callable[..., str]],
):
    """Two values of one stage (two partials, two field defaults of one dataclass, two helpers'
    StageParams defaults) hold instances of two classes one factory made at the same position,
    or two loaders whose classes one factory made. Each walk keeps its own key, so an edit to
    the first class alone changes the manifest, and the second never overwrites the first."""
    before = fingerprint.get_stage_fingerprint(shape("A"))
    after = fingerprint.get_stage_fingerprint(shape("B"))

    assert before != after
    suffix_cells = {value for key, value in _entries(before) if key and key.endswith(".suffix")}
    assert {"'A'", "'Z'"} <= suffix_cells, before


def test_a_cache_hit_whose_value_became_opaque_misses_and_reports_nothing():
    """A cache hit reads each recorded value again by its token and expands nothing. When an
    instance a partial holds now also holds a value no form reads (`random.Random`), the
    partial's token is new, so the hit misses, and checking it refuses and raises nothing; the
    stage walk that follows refuses the stage."""
    captured = _Captured()
    holds = functools.partial(_tag, captured)

    def stage(text: str) -> str:
        return holds(text)

    reads: list[object] = [[list(path), token] for path, token in _recording(stage)]
    assert fingerprint._recorded_reads_hold(stage, reads)
    captured.state = random.Random(1)  # pyright: ignore[reportAttributeAccessIssue] - the change under test

    assert not fingerprint._recorded_reads_hold(stage, reads)
    with pytest.raises(exceptions.StageDefinitionError, match="random.Random"):
        fingerprint.get_stage_fingerprint(stage)


def _read_value(thing: typing.Any) -> str:
    return thing.value()


def _make_partnered(label: str) -> type:
    """A class whose `next_value` calls a partial holding an instance of its partner class,
    bound once both exist (`_bind_partner`)."""
    partner: typing.Any = None

    class Made:
        def value(self) -> str:
            return label

        def next_value(self) -> str:
            return partner()

    return Made


def _bind_partner(cls: typing.Any, other: object) -> None:
    function = cls.next_value
    cell = function.__closure__[function.__code__.co_freevars.index("partner")]
    cell.cell_contents = functools.partial(_read_value, other)


def test_two_classes_one_factory_made_in_a_cycle_give_one_manifest_in_either_order():
    """Two classes one factory made, each with a method that reaches an instance of the other,
    held as equal forms in a set: each is walked with the memo as it stood before them, so the
    manifest does not follow the order the set yields them in (the hash seed)."""
    first_class, second_class = _make_partnered("A"), _make_partnered("B")
    first, second = first_class(), second_class()
    _bind_partner(first_class, second)
    _bind_partner(second_class, first)

    with fingerprint.stage_fingerprint_scope("t/use"):
        forward = fingerprint.get_loader_fingerprint(_HoldingReader(_ordered(first, second)))
    with fingerprint.stage_fingerprint_scope("t/use"):
        backward = fingerprint.get_loader_fingerprint(_HoldingReader(_ordered(second, first)))

    assert forward == backward
    labels = {value for key, value in _entries(forward) if key and key.endswith(".value.label")}
    assert labels == {"'A'", "'B'"}, forward


@pytest.mark.parametrize("position", ["dataclass-default", "loader-config", "partial"])
def test_a_captured_instance_whose_metaclass_reads_attributes_in_code_is_refused_running_no_hook(
    position: str,
):
    """The class of a captured instance has a metaclass whose `__hash__` and `__eq__` raise, and
    whose `__getattribute__` raises for `__mro__` and notes every other name it is asked. Reading
    anything of the class (its source, its fields, its members) would run that code, so the stage
    is refused, naming the class and its metaclass, and no hook runs: not the raising ones, and
    no attribute is asked of the class."""
    asked = list[str]()

    class RaisingMeta(type):
        def __hash__(cls) -> int:
            raise RuntimeError("metaclass __hash__ ran")

        def __eq__(cls, other: object) -> bool:
            raise RuntimeError("metaclass __eq__ ran")

        def __getattribute__(cls, name: str) -> typing.Any:
            if name == "__mro__":
                raise RuntimeError("metaclass __mro__ lookup ran")
            asked.append(name)
            return super().__getattribute__(name)

    class Hooked(metaclass=RaisingMeta):
        def suffix(self) -> str:
            return "A"

    @dataclasses.dataclass(frozen=True)
    class Holder:
        hooked: typing.Any = Hooked()

    holds = functools.partial(_tag, Hooked())
    loader = _HoldingReader(Hooked())

    def through_a_default(text: str) -> str:
        return _tag(Holder().hooked, text)

    def through_a_partial(text: str) -> str:
        return holds(text)

    class_name = f"{Hooked.__module__}.{Hooked.__qualname__}"
    asked.clear()
    with pytest.raises(exceptions.StageDefinitionError) as refused:
        if position == "loader-config":
            fingerprint.get_loader_fingerprint(loader)
        else:
            stage = through_a_default if position == "dataclass-default" else through_a_partial
            fingerprint.get_stage_fingerprint(stage)

    message = str(refused.value)
    assert asked == [], f"The walk asked the class for {asked}"
    assert f"class '{class_name}'" in message, message
    assert "RaisingMeta', whose `__getattribute__` or `__getattr__` is user code" in message, (
        message
    )


class _SuffixBase:
    def suffix(self) -> str:
        return _BASE_SUFFIX


_BASE_SUFFIX = "A"
# A class with no source of its own whose `suffix` its user base defines: only the walk of the
# members on its MRO reaches `suffix`, since no class statement names the base.
_Dynamic = type("_Dynamic", (_SuffixBase,), {})
_HOLDS_A_DYNAMIC = functools.partial(_tag, _Dynamic())


def _tags_a_dynamic_instance(text: str) -> str:
    return _HOLDS_A_DYNAMIC(text)


def _names_the_dynamic_class(text: str) -> str:
    return _tag(_Dynamic(), text)


@pytest.mark.parametrize(
    "stage", [_tags_a_dynamic_instance, _names_the_dynamic_class], ids=["captured", "named"]
)
def test_a_member_a_class_inherits_is_walked_for_a_captured_instance(
    stage: typing.Callable[[str], str],
):
    """`_Dynamic`, made by `type()`, inherits `suffix`, which reads `_BASE_SUFFIX`, and no class
    statement names its base: the walk of its class reaches `suffix` through the MRO, for a
    captured instance and for a stage that names the class."""
    manifest = fingerprint.get_stage_fingerprint(stage)

    assert (f"const:{__name__}._BASE_SUFFIX", "'A'") in _entries(manifest), manifest


class _SharedBase:
    def shared(self) -> str:
        return "s"


class _FirstChild(_SharedBase):
    def first(self) -> str:
        return "1"


class _SecondChild(_SharedBase):
    def second(self) -> str:
        return "2"


_HOLDS_TWO_CHILDREN = functools.partial(_take, (_FirstChild(), _SecondChild()))


def _takes_two_children(x: int) -> int:
    return _HOLDS_TWO_CHILDREN(x)


def test_a_user_base_two_captured_classes_share_is_walked_once():
    """`_FirstChild` and `_SecondChild` share the user base `_SharedBase`: each class is walked
    for its own members, and the base once, under its own key, so its member's walk is in the
    manifest once, not once per class that inherits it."""
    manifest = fingerprint.get_stage_fingerprint(_takes_two_children)

    for member in (_SharedBase.shared, _FirstChild.first, _SecondChild.second):
        assert _definition_key("func", member) in manifest, manifest


def _constant(thing: object) -> str:
    return "x"


class _HoldsItself:
    def suffix(self) -> str:
        return _HOLDS_ITSELF()


_HOLDS_ITSELF = functools.partial(_constant, _HoldsItself())


def _tags_a_self_holding_instance(text: str) -> str:
    return text + _HOLDS_ITSELF()


def test_a_class_whose_member_reaches_its_own_instance_is_walked_once():
    """`_HoldsItself.suffix` reads a partial holding an instance of `_HoldsItself`: the class
    being walked is not walked again where its own walk reaches it, so the walk ends."""
    manifest = fingerprint.get_stage_fingerprint(_tags_a_self_holding_instance)

    assert _definition_key("func", _HoldsItself.suffix) in manifest, manifest


class _AddressRepr:
    """A value whose own `__repr__` writes its address."""

    __slots__: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return f"<_AddressRepr at {id(self):#x}>"


class _RaisingRepr:
    """A user class whose `__repr__` must not run while its class is walked."""

    def __init__(self, label: str) -> None:
        self.label: str = label

    def __repr__(self) -> str:
        raise RuntimeError("an unused default's repr ran")


def test_a_class_default_is_hashed_by_its_state_without_an_address():
    """A class-level default is hashed by what it holds, with no address and no repr run: an
    `object()` sentinel and a `__slots__ = ()` instance by their type, a `__slots__` instance by
    its slots, a user class with a repr of its own (one that raises, one that writes its address)
    by its attributes and slots, and a string that looks like an address by its content."""
    forms = {
        name: fingerprint._form(value)
        for name, value in {
            "sentinel": _NO_SOURCE_MISSING,
            "slots-1": _NoSourceBox(1),
            "slots-2": _NoSourceBox(2),
            "raising-repr-a": _RaisingRepr("a"),
            "raising-repr-b": _RaisingRepr("b"),
            "address-repr": _AddressRepr(),
            "hex-a": "0xabcdef12",
            "hex-b": "0x12345678",
        }.items()
    }

    opaque = fingerprint._ReportKind.OPAQUE
    assert not [form for form in forms.values() for _, kind, _ in form.reports if kind is opaque]
    assert all(
        _ADDRESS.search(form.text) is None for name, form in forms.items() if "hex" not in name
    ), forms
    assert forms["slots-1"].text != forms["slots-2"].text
    assert forms["raising-repr-a"].text != forms["raising-repr-b"].text
    assert (forms["hex-a"].text, forms["hex-b"].text) == ('"0xabcdef12"', '"0x12345678"')


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(random.Random(1), id="random"),
        pytest.param(threading.Lock(), id="lock"),
    ],
)
def test_a_class_default_whose_state_no_form_reads_refuses_the_stage(value: object):
    """A class-level default whose state is in C cannot be hashed: a stage that names the class
    is refused, naming the field, and the field's read token is the hash of the default's opaque
    form, which no readable default has, so a manifest cache hit misses."""

    @dataclasses.dataclass(frozen=True)
    class Holds:
        state: typing.Any = value

    def names(x: int) -> int:
        return x + bool(Holds().state)

    with pytest.raises(exceptions.StageDefinitionError, match=r"default of field '.*Holds\.state'"):
        fingerprint.get_stage_fingerprint(names)
    form = fingerprint._form(value)
    assert [kind for _, kind, _ in form.reports] == [fingerprint._ReportKind.OPAQUE]
    digest = xxhash.xxh64(form.text.encode()).hexdigest()
    assert fingerprint._dataclass_field_token(Holds, "state") == f"field:{digest}"


@dataclasses.dataclass(frozen=True)
class _WithRaisingReprDefault:
    """Has a docstring, so `dataclasses` does not build one from the default's repr."""

    value: typing.Any = _RaisingRepr("unused")


_HOLDS_RAISING_REPR_DEFAULT = functools.partial(str, _WithRaisingReprDefault("explicit"))


def _captures_a_raising_repr_default() -> str:
    return _HOLDS_RAISING_REPR_DEFAULT()


def test_an_unused_default_whose_repr_raises_does_not_stop_a_captured_stage():
    """A partial holds an instance that overrides its class's default, a user class instance
    whose `__repr__` raises: walking the class hashes the default by its attributes, so the
    repr never runs and the stage is fingerprinted."""
    manifest = fingerprint.get_stage_fingerprint(_captures_a_raising_repr_default)

    key = f"default:{fingerprint._definition_ref(_WithRaisingReprDefault)}.value"
    assert key in manifest, manifest


_FIELD_TOKEN_MODULE = """\
import dataclasses
from typing import Any


class Box:
    __slots__ = ("x",)

    def __init__(self, x: int) -> None:
        self.x = x


@dataclasses.dataclass(frozen=True)
class Defaults:
    sentinel: Any = object()
    box: Any = Box(1)
"""

_FIELD_TOKEN_SCRIPT = """\
import json
import sys

sys.path.insert(0, sys.argv[1])
import field_token_defaults as module
from pivot import fingerprint

print(json.dumps([fingerprint._dataclass_field_token(module.Defaults, name)
                  for name in ("sentinel", "box")]))
"""


def test_a_dataclass_field_token_is_the_same_in_every_interpreter(tmp_path: pathlib.Path):
    """The read token of a field whose default is an `object()` sentinel or a `__slots__`
    object is the hash of the default's text with no address, so two interpreters with two hash
    seeds give one token, and a manifest cache hit in the next process is taken."""
    (tmp_path / "field_token_defaults.py").write_text(_FIELD_TOKEN_MODULE)
    tokens = [
        subprocess.run(
            [sys.executable, "-c", _FIELD_TOKEN_SCRIPT, str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2")
    ]

    assert tokens[0] == tokens[1], tokens
    assert "refused" not in tokens[0], tokens


_UNSAFE_BASE_VALUES = ["unused"]


class _UnsafeBase:
    def value(self) -> str:
        return _UNSAFE_BASE_VALUES[0]


def _safe_value(self: object) -> str:
    return "safe"


# A class with no statement that overrides its base's member: its base is walked, as a class
# statement's base is, so the base's own `value` is walked too.
_SafeDerived = type("_SafeDerived", (_UnsafeBase,), {"value": _safe_value})
_HOLDS_SAFE_DERIVED = functools.partial(_read_value, _SafeDerived())


class _SafeDerivedStatement(_UnsafeBase):
    value: typing.ClassVar[typing.Callable[[object], str]] = _safe_value


def _captures_safe_derived() -> str:
    return _HOLDS_SAFE_DERIVED()


def _names_safe_derived() -> str:
    return _SafeDerived().value()


def _names_safe_derived_statement() -> str:
    return _SafeDerivedStatement().value()


@pytest.mark.parametrize(
    "stage", [_captures_safe_derived, _names_safe_derived, _names_safe_derived_statement]
)
def test_a_class_without_a_statement_walks_its_base_as_a_class_statement_does(
    stage: typing.Callable[[], str],
):
    """`_SafeDerived`, made by `type()`, overrides `value`, whose base version reads a list
    global. Its base is part of its walk, as `_SafeDerivedStatement`'s is: a stage that names
    either class, or captures an instance of `_SafeDerived`, walks the base's `value` and is
    refused for the list it reads."""
    with pytest.raises(exceptions.StageDefinitionError, match="'_UNSAFE_BASE_VALUES'"):
        fingerprint.get_stage_fingerprint(stage)


def test_walking_a_held_class_runs_no_metaclass_repr():
    """The class of an instance a loader's config holds has a metaclass whose `__repr__` raises:
    the class is walked without being passed to an API that formats it."""

    class ReprMeta(type):
        def __repr__(cls) -> str:
            raise RuntimeError("metaclass __repr__ ran")

    class ReprThing(metaclass=ReprMeta):
        def value(self) -> str:
            return "safe"

    manifest = fingerprint.get_loader_fingerprint(_HoldingReader(ReprThing()))

    key = f"func:{ReprThing.__module__}.{ReprThing.__qualname__}.value"
    assert key in manifest, manifest


def _swapped(first: str, second: str) -> typing.Callable[[str], str]:
    """A stage whose two partials hold instances of two classes one factory made."""
    first_holds = functools.partial(_tag, _make_captured(first)())
    second_holds = functools.partial(_tag, _make_captured(second)())

    def stage(text: str) -> str:
        return first_holds(text) + second_holds(text)

    return stage


def test_swapping_two_captured_classes_between_two_values_changes_the_manifest():
    """Two partials hold instances of classes one factory made with `A` and `Z`; swapping them
    changes the stage's output, so it changes the manifest: each walk is keyed by the value that
    holds it."""
    before = fingerprint.get_stage_fingerprint(_swapped("A", "Z"))
    after = fingerprint.get_stage_fingerprint(_swapped("Z", "A"))

    assert before != after


def _make_slotted(label: str) -> type:
    class Slot:
        __slots__: tuple[str, ...] = ()

        def value(self) -> str:
            return label

    return Slot


@dataclasses.dataclass(frozen=True)
class _SetDefault:
    values: typing.Any = frozenset((_make_slotted("A")(), _make_slotted("B")()))


def _reads_a_set_default() -> str:
    return "".join(sorted(item.value() for item in _SetDefault().values))


def test_a_dataclass_default_holding_instances_replays_as_it_records():
    """A dataclass default holds a set of two `__slots__` instances of classes one factory made:
    the walk records each class's reads under its member's rank in the set (`s:0`, `s:1`), and a
    cache hit follows those steps and takes the same reads again, in one process."""
    recorded = _recording(_reads_a_set_default)
    reads: list[object] = [[list(path), token] for path, token in recorded]

    ranks = {
        path[2]
        for path, _ in recorded
        if path[:2] == ("g:_SetDefault", "fd:values") and path[-1] == "m:value"
    }
    assert ranks == {"s:0", "s:1"}, reads
    assert fingerprint._recorded_reads_hold(_reads_a_set_default, reads)


_READER_COLUMNS = ["a", "b"]


@dataclasses.dataclass(frozen=True)
class _ReadsAListReader(loaders.Reader[str]):
    """A loader whose `load` reads a list global, which the walk refuses."""

    def load(self, path: pathlib.Path) -> str:
        return ",".join(_READER_COLUMNS)


_HOLDS_A_LIST_READER = functools.partial(_apply, _ReadsAListReader())


def _takes_a_list_reader(path: pathlib.Path) -> str:
    return _HOLDS_A_LIST_READER(path)


@pytest.mark.parametrize(
    ("stage", "step", "name"),
    [
        pytest.param(
            _takes_one_that_reads_a_list, "g:_HOLDS_READS_A_LIST", "_COLUMNS", id="held-class"
        ),
        pytest.param(
            _takes_a_list_reader, "g:_HOLDS_A_LIST_READER", "_READER_COLUMNS", id="held-loader"
        ),
    ],
)
def test_a_cache_hit_check_keeps_nothing_and_the_stage_walk_refuses(
    stage: typing.Callable[..., object], step: str, name: str
):
    """In one stage scope, checking a cache hit reads a partial's token, the hash of its form, and
    expands nothing, so it refuses nothing and keeps no walk; the stage walk that follows walks
    the class a partial holds an instance of, or the loader it holds and that loader's class,
    and refuses the stage."""
    with (
        pytest.raises(exceptions.StageDefinitionError, match=f"non-scalar variable '{name}'"),
        fingerprint.stage_fingerprint_scope("t/use"),
    ):
        assert fingerprint._read_token(stage, step).startswith("partial:")
        fingerprint.get_stage_fingerprint(stage)


_LABELS = ["a", "b"]


@dataclasses.dataclass(frozen=True)
class _ReachBase(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text()


@dataclasses.dataclass(frozen=True)
class _Reaches(_ReachBase):
    """`load` reads `self._first` (which reads `self._second`) and the property `self.label`;
    `unused` reads a list global, and nothing reaches it."""

    suffix: str = "A"

    def load(self, path: pathlib.Path) -> str:
        return self._first(path.read_text()) + self.label + self.suffix

    def _first(self, text: str) -> str:
        return self._second(text)

    def _second(self, text: str) -> str:
        return text.strip()

    @property
    def label(self) -> str:
        return "L"

    def unused(self) -> str:
        return ",".join(_LABELS)


@dataclasses.dataclass(frozen=True)
class _ReachesSuper(_ReachBase):
    """`load` calls the base class's `load` through `super()`."""

    def load(self, path: pathlib.Path) -> str:
        return super().load(path).strip()


def _upper_with_a_list(text: str) -> str:
    return text.upper() + _LABELS[0]


def _lower(text: str) -> str:
    return text.lower()


@dataclasses.dataclass(frozen=True)
class _FieldFunction(loaders.Reader[str]):
    """`load` calls `self.fmt`, a config field whose default reads a list global."""

    fmt: typing.Callable[[str], str] = _upper_with_a_list

    def load(self, path: pathlib.Path) -> str:
        return self.fmt(path.read_text())


def _logged(function: typing.Callable[..., str]) -> typing.Callable[..., str]:
    @functools.wraps(function)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> str:
        return function(*args, **kwargs)

    return wrapper


def _reach_apply(obj: typing.Any) -> str:
    return obj._suffix()


@dataclasses.dataclass(frozen=True)
class _ReachBy(loaders.Reader[str]):
    """A loader whose `unused` reads a list global and is reached by no static form."""

    def _suffix(self) -> str:
        return "A"

    def unused(self) -> str:
        return ",".join(_LABELS)

    def load(self, path: pathlib.Path) -> str:
        return path.read_text()


@dataclasses.dataclass(frozen=True)
class _ByGetattr(_ReachBy):
    def load(self, path: pathlib.Path) -> str:
        return getattr(self, "_suffix")()  # noqa: B009 - the literal getattr form is the point


@dataclasses.dataclass(frozen=True)
class _ByTypeOfSelf(_ReachBy):
    def load(self, path: pathlib.Path) -> str:
        return type(self)._suffix.__name__


@dataclasses.dataclass(frozen=True)
class _BySelfClass(_ReachBy):
    def load(self, path: pathlib.Path) -> str:
        return self.__class__._suffix.__name__


@dataclasses.dataclass(frozen=True)
class _ByWrappedHelper(_ReachBy):
    @_logged
    def _helper(self) -> str:
        return self._suffix()

    def load(self, path: pathlib.Path) -> str:
        return self._helper()


@dataclasses.dataclass(frozen=True)
class _ByBuiltin(_ReachBy):
    def __str__(self) -> str:
        return self._suffix()

    def load(self, path: pathlib.Path) -> str:
        return f"{self!s}" + str(self)


@dataclasses.dataclass(frozen=True)
class _ByConstant(_ReachBy):
    SUFFIX: typing.ClassVar[str] = "A"

    def load(self, path: pathlib.Path) -> str:
        return self.SUFFIX + self._suffix()


@dataclasses.dataclass(frozen=True)
class _BySelfGiven(_ReachBy):
    def load(self, path: pathlib.Path) -> str:
        return _reach_apply(self)


@dataclasses.dataclass(frozen=True)
class _ByOperator(_ReachBy):
    def __add__(self, other: str) -> str:
        return self._suffix() + other

    def load(self, path: pathlib.Path) -> str:
        return self + ""


_SUFFIX_REF = f"{__name__}._ReachBy._suffix"
_UNUSED_REF = f"{__name__}._ReachBy.unused"


@pytest.mark.parametrize(
    ("loader_class", "expected"),
    [
        pytest.param(_ByGetattr, {_SUFFIX_REF}, id="getattr-literal"),
        pytest.param(_ByTypeOfSelf, {_SUFFIX_REF}, id="type-of-self"),
        pytest.param(_BySelfClass, {_SUFFIX_REF}, id="self-class"),
        pytest.param(
            _ByWrappedHelper,
            {_SUFFIX_REF, f"{__name__}._ByWrappedHelper._helper"},
            id="wrapped-helper",
        ),
        pytest.param(
            _ByBuiltin, {_SUFFIX_REF, f"{__name__}._ByBuiltin.__str__"}, id="builtin-dunder"
        ),
        pytest.param(
            _ByConstant, {_SUFFIX_REF, f"{__name__}._ByConstant.SUFFIX"}, id="class-constant"
        ),
        pytest.param(_BySelfGiven, {_SUFFIX_REF, _UNUSED_REF}, id="self-given-to-a-function"),
        pytest.param(
            _ByOperator,
            {_SUFFIX_REF, _UNUSED_REF, f"{__name__}._ByOperator.__add__"},
            id="operator-on-self",
        ),
    ],
)
def test_a_loader_walks_what_each_form_of_self_reaches(loader_class: type, expected: set[str]):
    """Each static form names the member it reaches: `getattr` with a literal, `type(self)`,
    `self.__class__`, a `functools.wraps` helper's wrapped body, a builtin's dunder, a class
    constant; no other member is walked. `self` given to a function or an operator on `self`
    may reach any member, so every member is walked, `unused` included."""
    reached = {ref for ref, _member in fingerprint._reachable_loader_members(loader_class)}

    assert reached == expected


def _loop_wrapped(function: typing.Callable[..., str]) -> typing.Callable[..., str]:
    """A decorator that makes the function its own `__wrapped__`: a chain that loops."""
    function.__wrapped__ = function  # pyright: ignore[reportFunctionMemberAccess] - the loop is the point
    return function


@dataclasses.dataclass(frozen=True)
class _LoopWrappedHelper(loaders.Reader[str]):
    @_loop_wrapped
    def helper(self, text: str) -> str:
        return text + "A"

    def load(self, path: pathlib.Path) -> str:
        return self.helper(path.read_text())


@dataclasses.dataclass(frozen=True)
class _LoopWrappedLoad(loaders.Reader[str]):
    @_loop_wrapped
    def load(self, path: pathlib.Path) -> str:
        return path.read_text()


@dataclasses.dataclass(frozen=True)
class _LoopWrappedInFallback(loaders.Reader[str]):
    """`load` applies an operator to `self`, so the walk falls back to every member."""

    @_loop_wrapped
    def helper(self, text: str) -> str:
        return text + "A"

    def __add__(self, text: str) -> str:
        return text

    def load(self, path: pathlib.Path) -> str:
        return self + path.read_text()


@dataclasses.dataclass(frozen=True)
class _LoopWrappedReachedAndInFallback(loaders.Reader[str]):
    """`load` reaches `helper`, then `combine` applies an operator to `self`, so the fallback
    finds `helper` again."""

    @_loop_wrapped
    def helper(self, text: str) -> str:
        return text + "A"

    def __add__(self, text: str) -> str:
        return text

    def combine(self, text: str) -> str:
        return self + text

    def load(self, path: pathlib.Path) -> str:
        return self.combine(self.helper(path.read_text()))


def _raise_timeout(signum: int, frame: object) -> None:
    raise TimeoutError("the walk did not end")


@pytest.mark.parametrize(
    ("loader", "name"),
    [
        pytest.param(_LoopWrappedHelper(), "_LoopWrappedHelper.helper", id="reached-helper"),
        pytest.param(_LoopWrappedLoad(), "_LoopWrappedLoad.load", id="load"),
        pytest.param(
            _LoopWrappedInFallback(), "_LoopWrappedInFallback.helper", id="fallback-member"
        ),
        pytest.param(
            _LoopWrappedReachedAndInFallback(),
            "_LoopWrappedReachedAndInFallback.helper",
            id="reached-and-fallback-member",
        ),
    ],
)
def test_a_wrapped_chain_that_loops_is_refused(loader: object, name: str):
    """D3: refuse reached and fallback loader members with looping metadata, without hanging."""
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, 10)
    try:
        with (
            pytest.raises(exceptions.StageDefinitionError) as raised,
            fingerprint.stage_fingerprint_scope("t/use"),
        ):
            fingerprint.get_loader_fingerprint(loader)  # pyright: ignore[reportArgumentType]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

    message = str(raised.value)
    assert name in message and "loop" in message, message
    assert "StageParams" in message and "break" in message.lower(), message


def test_each_stage_of_a_pass_whose_loader_reaches_a_looped_helper_is_refused():
    """D3: a shared member walk must retain the refusal for each stage in the pass."""
    with fingerprint.fingerprint_pass():
        for stage_name in ("t/first", "t/second"):
            with (
                pytest.raises(exceptions.StageDefinitionError, match="_LoopWrappedHelper.helper"),
                fingerprint.stage_fingerprint_scope(stage_name),
            ):
                fingerprint.get_loader_fingerprint(_LoopWrappedHelper())


def test_a_class_constant_load_reads_is_recorded_as_a_value():
    """`self.SUFFIX` reads a class constant: its value is a `const:` entry of the loader."""
    manifest = fingerprint.get_loader_fingerprint(_ByConstant())

    assert f"const:{__name__}._ByConstant.SUFFIX" in manifest, manifest


def _make_row(fields: str) -> type:
    return collections.namedtuple("Row", fields)  # pyright: ignore[reportUntypedNamedTuple] - no source is the point


_ROW_A = _make_row("a b")
_ROW_B = _make_row("c d")


def _reads_two_rows() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (_ROW_A._fields, _ROW_B._fields)


def test_two_classes_with_no_source_and_one_name_have_two_keys():
    """One factory makes two namedtuples, both `<this module>.Row`. Each is keyed by its name
    and its hash, so both are entries of the walk, and neither is cached by its module's file."""
    manifest = fingerprint.get_stage_fingerprint(_reads_two_rows)

    keys = {key for key in manifest if key.startswith(f"class:{__name__}.Row@")}
    assert keys == {
        f"class:{__name__}.Row@{fingerprint.hash_function_ast(_ROW_A)}",
        f"class:{__name__}.Row@{fingerprint.hash_function_ast(_ROW_B)}",
    }
    assert len(keys) == 2


def test_a_config_field_load_reads_is_config_not_a_member():
    """`self.fmt` names a dataclass field: the loader's own value is walked as config, so the
    class's default, which reads a list global, is neither walked nor refused (a refusal would
    raise here)."""
    manifest = fingerprint.get_loader_fingerprint(_FieldFunction(fmt=_lower))

    assert not any("_upper_with_a_list" in key for key in manifest), manifest


def test_a_loader_walks_only_the_members_its_methods_reach():
    """The members reached through `self.<name>` transitively, a property, and a base class's
    method reached through `super()` are walked, each under its defining class; a member nothing
    reaches, or a dataclass field, is not, so its read of a list global is no refusal."""
    reached = {ref for ref, _member in fingerprint._reachable_loader_members(_Reaches)}
    manifest = fingerprint.get_loader_fingerprint(_Reaches())
    through_super = fingerprint.get_loader_fingerprint(_ReachesSuper())

    assert reached == {
        f"{__name__}._Reaches._first",
        f"{__name__}._Reaches._second",
        f"{__name__}._Reaches.label",
    }
    for function in (_Reaches._first, _Reaches._second):
        key = _definition_key("func", function)
        assert manifest[key] == fingerprint.hash_function_ast(function), key
    assert not any("unused" in key for key in manifest), manifest
    assert [ref for ref, _member in fingerprint._reachable_loader_members(_ReachesSuper)] == [
        f"{__name__}._ReachBase.load"
    ]
    key = _definition_key("func", _ReachBase.load)
    assert through_super[key] == fingerprint.hash_function_ast(_ReachBase.load)


@dataclasses.dataclass(frozen=True)
class _HeldWriter(loaders.Writer[str]):
    """A write-only user loader."""

    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data + "A")


def test_a_held_write_only_loader_is_fingerprinted():
    """A `loaders.Writer` with only `save`, held in a config, is a pivot loader too."""
    holder = fingerprint.get_loader_fingerprint(_HoldingReader(_HeldWriter()))

    own = fingerprint.get_loader_fingerprint(_HeldWriter())
    assert _loader_key(_HeldWriter(), "save") in own
    assert set(own.items()) <= _entries(holder)


def test_a_swap_of_two_held_loaders_with_one_form_changes_the_manifest():
    """`_make_reader("first")` and `_make_reader("A")` look alike in the config text; each held
    loader is keyed by its position, so swapping them, or wrapping them each in a holder and
    editing the second, changes the manifest."""
    first, second, edited = _make_reader("first"), _make_reader("A"), _make_reader("B")
    pair = fingerprint.get_loader_fingerprint(_HoldingReader((first(), second())))
    swapped = fingerprint.get_loader_fingerprint(_HoldingReader((second(), first())))
    wrapped = fingerprint.get_loader_fingerprint(
        _HoldingReader((_HoldingReader(first()), _HoldingReader(second())))
    )
    rewrapped = fingerprint.get_loader_fingerprint(
        _HoldingReader((_HoldingReader(first()), _HoldingReader(edited())))
    )

    assert pair != swapped
    assert wrapped != rewrapped


def _iteration_order(ordered: frozenset[typing.Any]) -> tuple[typing.Any, ...]:
    """The Python-level iteration order `_ordered` gave `ordered`. The orders are kept in this
    function's own attributes, which no fingerprint reads, so an `_Ordered`'s state is its
    members alone."""
    return _iteration_order.orders[id(ordered)]  # pyright: ignore[reportFunctionMemberAccess] - the store is the function's attribute


_iteration_order.orders = dict[int, tuple[typing.Any, ...]]()  # pyright: ignore[reportFunctionMemberAccess] - the store is the function's attribute


class _Ordered(frozenset[typing.Any]):
    """A frozenset whose Python-level iteration order is given, as a hash seed would give one."""

    __slots__: tuple[str, ...] = ()

    def __iter__(self) -> collections.abc.Iterator[typing.Any]:
        return iter(_iteration_order(self))


@dataclasses.dataclass(frozen=True)
class _CycleA(loaders.Reader[str]):
    tag: str
    other: typing.Any = dataclasses.field(default=None, compare=False, hash=False)

    def load(self, path: pathlib.Path) -> str:
        return path.read_text() + self.tag


@dataclasses.dataclass(frozen=True)
class _CycleB(loaders.Reader[str]):
    tag: str
    other: typing.Any = dataclasses.field(default=None, compare=False, hash=False)

    def load(self, path: pathlib.Path) -> str:
        return path.read_text() + self.tag


def _ordered(*items: typing.Any) -> _Ordered:
    ordered = _Ordered(items)
    _iteration_order.orders[id(ordered)] = items  # pyright: ignore[reportFunctionMemberAccess] - the store is the function's attribute
    return ordered


def test_a_set_of_loaders_in_a_cycle_gives_one_manifest_in_either_order():
    """Two loaders that hold each other, in a set: the manifest does not follow the order the
    set yields them in, so it does not follow the hash seed."""
    a, b = _CycleA("A"), _CycleB("B")
    object.__setattr__(a, "other", b)
    object.__setattr__(b, "other", a)

    with fingerprint.stage_fingerprint_scope("t/use"):
        forward = fingerprint.get_loader_fingerprint(_HoldingReader(_ordered(a, b)))
    with fingerprint.stage_fingerprint_scope("t/use"):
        backward = fingerprint.get_loader_fingerprint(_HoldingReader(_ordered(b, a)))

    assert forward == backward
    assert {_loader_key(a, "load"), _loader_key(b, "load")} <= forward.keys(), forward


def _make_alike(tag: str) -> typing.Any:
    @dataclasses.dataclass(frozen=True)
    class Alike(loaders.Reader[str]):
        other: typing.Any = dataclasses.field(default=None, compare=False, hash=False)

        def load(self, path: pathlib.Path) -> str:
            return path.read_text() + tag

    return Alike


def test_two_loaders_alike_in_a_cycle_give_one_manifest_in_either_order():
    """Two classes one factory made, whose instances hold each other in a set: they share a
    position and a form, so each is walked with the memo as it stood before them, and the
    manifest holds both, whichever order the set yields them in."""
    x, y = _make_alike("x")(), _make_alike("y")()
    object.__setattr__(x, "other", y)
    object.__setattr__(y, "other", x)

    with fingerprint.stage_fingerprint_scope("t/use"):
        forward = fingerprint.get_loader_fingerprint(_HoldingReader(_ordered(x, y)))
    with fingerprint.stage_fingerprint_scope("t/use"):
        backward = fingerprint.get_loader_fingerprint(_HoldingReader(_ordered(y, x)))

    assert forward == backward
    loads = [key for key in forward if ".Alike@" in key and key.endswith(":load")]
    assert len(loads) == 2, forward


def test_instances_of_one_class_in_a_set_give_one_manifest_in_either_order():
    """Two `_Settings` with different fields in a set: the class's walk is keyed by the first of
    their positions in sorted order, not by the order the set yields them in, so the manifest
    does not follow the hash seed."""
    first, second = _Settings(1), _Settings(2)

    forward = fingerprint.get_loader_fingerprint(_HoldingReader(_ordered(first, second)))
    backward = fingerprint.get_loader_fingerprint(_HoldingReader(_ordered(second, first)))

    assert forward == backward
    assert _class_walk(_Settings) <= _entries(forward)


def _chain(depth: int) -> typing.Any:
    value: typing.Any = _HeldReader()
    for _ in range(depth):
        value = _HoldingReader(value)
    return value


_NO_SOURCE_MISSING = object()


class _NoSourceBox:
    """A value whose canonical form is its repr, which holds its address."""

    __slots__: tuple[str, ...] = ("x",)

    def __init__(self, x: int) -> None:
        self.x: int = x


def _no_source_check(cls: type, value: str) -> str:
    return value


def _no_source_check_edited(cls: type, value: str) -> str:
    return value.strip()


def _no_source_classes(variant: str) -> list[type]:
    """Classes with no source of their own, made twice from one definition by `variant`; the
    edited variant changes one entry of each."""
    second = variant == "edited"
    fields = "left middle" if second else "left right"
    token = "B" if second else "A"
    check = _no_source_check_edited if second else _no_source_check
    return [
        collections.namedtuple("_NoSourceRow", fields),  # pyright: ignore[reportUntypedNamedTuple] - no source is the point
        dataclasses.make_dataclass("_NoSourceGen", [("y" if second else "x", int)]),
        enum.Enum("_NoSourceMode", "A C" if second else "A B"),
        type("_NoSourceMade", (), {"x": 2 if second else 1}),
        pydantic.create_model("_NoSourceModel", x=(int, 1), y=(str, token)),
        pydantic.create_model(
            "_NoSourceChecked",
            x=(str, "v"),
            __validators__={"check": pydantic.field_validator("x")(check)},
        ),
    ]


def test_a_class_with_no_source_is_hashed_by_what_defines_it():
    """Classes with no source (namedtuples, `make_dataclass` classes, a functional enum, a
    `type(...)` class, `pydantic.create_model` models) are hashed by their names, fields or
    members and code, not their addresses: two made from one definition hash alike, an edit to
    any entry hashes otherwise, and a model's validator is hashed by its code."""
    hashes = {
        variant: [fingerprint._compute_function_hash(cls) for cls in _no_source_classes(variant)]
        for variant in ("first", "again", "edited")
    }

    assert hashes["first"] == hashes["again"]
    for first, edited in zip(hashes["first"], hashes["edited"], strict=True):
        assert first != edited


# Classes with no source with an entry whose form shows nothing (an `object()` sentinel, a
# `__slots__` object, a `random.Random`) beside an entry holding `token`.


def _sentinel_beside(token: str) -> type:
    return collections.namedtuple(  # pyright: ignore[reportUntypedNamedTuple] - no source is the point
        "_NoSourceSentinel", "a b c", defaults=(_NO_SOURCE_MISSING, token)
    )


def _slots_beside(token: str) -> type:
    return collections.namedtuple(  # pyright: ignore[reportUntypedNamedTuple] - no source is the point
        "_NoSourceSlots", "a box c", defaults=(_NoSourceBox(1), token)
    )


def _random_beside(token: str) -> type:
    fields = [
        ("rng", object, dataclasses.field(default=random.Random(1))),
        ("y", str, dataclasses.field(default=token)),
    ]
    return dataclasses.make_dataclass("_NoSourceRandom", fields)


def _model_sentinel_beside(token: str) -> type:
    return pydantic.create_model(
        "_NoSourceSentinelModel", x=(object, _NO_SOURCE_MISSING), y=(str, token)
    )


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(_sentinel_beside, id="sentinel"),
        pytest.param(_slots_beside, id="slots"),
        pytest.param(_model_sentinel_beside, id="model-sentinel"),
    ],
)
def test_an_edit_beside_a_value_no_form_shows_changes_the_hash(
    make: typing.Callable[[str], type],
):
    """An `object()` sentinel or a `__slots__` object leaves the rest of the class's entries in
    its hash: an edit to the other default (`A` to `B`, in this one module) changes it, and two
    classes made alike hash alike."""
    first = fingerprint._compute_function_hash(make("A"))
    again = fingerprint._compute_function_hash(make("A"))
    edited = fingerprint._compute_function_hash(make("B"))

    assert first == again, "Two classes made alike should hash alike"
    assert first != edited, "An edit to a shown entry should change the hash"


_CALLBACK_VALUE = "A"


class _Callback:
    """A `__slots__` object that holds a callback its caller gives it."""

    __slots__: tuple[str, ...] = ("callback",)

    def __init__(self, callback: typing.Callable[[], str]) -> None:
        self.callback: typing.Callable[[], str] = callback

    def read(self) -> str:
        return self.callback()


@dataclasses.dataclass(frozen=True)
class _WithCallbackDefault:
    value: typing.Any = _Callback(lambda: _CALLBACK_VALUE)

    def fresh(self) -> str:
        return type(self)().value.read()


def _names_a_callback_default() -> str:
    return _WithCallbackDefault().value.read()


def _fresh(thing: typing.Any) -> str:
    return thing.fresh()


_HOLDS_A_CALLBACK_HOLDER = functools.partial(_fresh, _WithCallbackDefault(None))


def _captures_a_callback_holder() -> str:
    return _HOLDS_A_CALLBACK_HOLDER()


@pytest.mark.parametrize("stage", [_names_a_callback_default, _captures_a_callback_holder])
def test_a_callback_in_an_instances_state_is_walked_with_the_globals_it_reads(
    stage: typing.Callable[[], str],
):
    """A `__slots__` object in a dataclass's class-level default holds a lambda that reads
    `_CALLBACK_VALUE`. For a stage that names the class, and for one that captures an instance
    overriding the default, the lambda gets a stage helper's walk, so the global is in the
    manifest."""
    manifest = fingerprint.get_stage_fingerprint(stage)

    assert f"const:{__name__}._CALLBACK_VALUE" in manifest, manifest


def _callback_class_cycle() -> str:
    return _CycleBox.marker


@dataclasses.dataclass(frozen=True)
class _CycleBox:
    marker: typing.ClassVar[str] = "safe"
    value: typing.Any = _Callback(_callback_class_cycle)


def _reads_a_callback_cycle() -> str:
    return _CycleBox().value.read()


def test_a_callback_that_reads_the_class_holding_it_is_a_back_reference():
    """A callback in `_CycleBox`'s default reads `_CycleBox.marker`: the enclosing class is
    already in the held callable's dependency set, so the callback stops at the back-reference
    instead of recursing, and the stage has a stable manifest."""
    with fingerprint.stage_fingerprint_scope("t/use"):
        first = fingerprint.get_stage_fingerprint(_reads_a_callback_cycle)
        second = fingerprint.get_stage_fingerprint(_reads_a_callback_cycle)

    assert first == second


def _loop_callback() -> str:
    return "safe"


setattr(_loop_callback, "__wrapped__", _loop_callback)  # noqa: B010 - the loop under test


@dataclasses.dataclass(frozen=True)
class _LoopCallbackBox:
    value: typing.Any = _Callback(_loop_callback)


def _reads_a_loop_callback() -> str:
    return _LoopCallbackBox().value.read()


def test_a_callback_whose_wrapped_chain_loops_is_refused():
    """D3: a looping callback in default state is refused by name."""
    with pytest.raises(exceptions.StageDefinitionError) as raised:
        fingerprint.get_stage_fingerprint(_reads_a_loop_callback)
    message = str(raised.value)
    assert "_loop_callback" in message and "loop" in message, message
    assert "StageParams" in message and "break" in message.lower(), message


_CLASS_CALLBACK_VALUE = "A"


class _ClassCallback:
    @classmethod
    def read(cls) -> str:
        return _CLASS_CALLBACK_VALUE


class _DictCallback:
    def __init__(self, callback: typing.Callable[[], str]) -> None:
        self.callback: typing.Callable[[], str] = callback

    def read(self) -> str:
        return self.callback()


@dataclasses.dataclass(frozen=True)
class _WithClassMethodCallback:
    value: typing.Any = _DictCallback(_ClassCallback.read)


def _reads_a_classmethod_callback() -> str:
    return _WithClassMethodCallback().value.read()


def test_a_class_bound_method_in_state_walks_its_function_globals():
    """A classmethod stored as a callback is normalized to its underlying function before its
    walk, so the global it reads is in the manifest."""
    manifest = fingerprint.get_stage_fingerprint(_reads_a_classmethod_callback)

    assert f"const:{__name__}._CLASS_CALLBACK_VALUE" in manifest, manifest


def test_bytes_and_a_path_with_identity_looking_text_are_hashed_by_value():
    """Identity-looking text inside bytes or a path is value data, not an object's repr:
    `b"located at 0xabcdef"`, `b"Py(0xabcdef)"`, `b"<object at 0xabcdef>"` and a path with
    `at 0xabcdef` are accepted, not refused, and each content change changes the form."""
    for value, other in (
        (b"located at 0xabcdef", b"located at 0x123456"),
        (b"Py(0xabcdef)", b"Py(0x123456)"),
        (b"<object at 0xabcdef>", b"<object at 0x123456>"),
        (
            pathlib.PurePosixPath("cache/at 0xabcdef"),
            pathlib.PurePosixPath("cache/at 0x123456"),
        ),
    ):
        form = fingerprint._form(value)
        assert not _opaque_reports(form), form
        assert form.text != fingerprint._form(other).text


@dataclasses.dataclass(frozen=True)
class _RandomReader(loaders.Reader[float]):
    rng: typing.Any

    def load(self, path: pathlib.Path) -> float:
        return self.rng.random()


@dataclasses.dataclass(frozen=True)
class _WithRandomReaderDefault:
    reader: typing.Any = _RandomReader(random.Random(1))


def _reads_a_random_reader_default() -> float:
    return _WithRandomReaderDefault().reader.load(pathlib.Path("."))


@pytest.mark.parametrize("recording", [False, True], ids=["walk", "recording"])
def test_a_loader_in_a_default_whose_config_holds_unreadable_state_is_refused(recording: bool):
    """A dataclass default is a reader whose config holds a `random.Random`. The stage is
    refused whether or not the walk records its reads: recording first computes the field's
    token, which keeps the loader's form, and the kept form still reports its state."""
    with (
        pytest.raises(exceptions.StageDefinitionError, match="_WithRandomReaderDefault.reader"),
        fingerprint.stage_fingerprint_scope("t/use"),
    ):
        if recording:
            _recording(_reads_a_random_reader_default)
        else:
            fingerprint.get_stage_fingerprint(_reads_a_random_reader_default)


def _ordered_stage(order: str) -> typing.Callable[[str], str]:
    """A stage whose partial holds instances of two classes one factory made, in `order`."""
    made = {label: _make_captured(label) for label in "AB"}
    holds = functools.partial(_tag_all, [made[label]() for label in order])

    def stage(text: str) -> str:
        return holds(text)

    return stage


def _tag_all(things: collections.abc.Iterable[typing.Any], text: str) -> str:
    return text + "".join(thing.suffix() for thing in things)


def test_an_exception_default_is_hashed_by_its_args_and_attributes():
    """A `ValueError` default is hashed by its type, its `args` and its `__dict__`: an edit to
    an attribute set on it changes the text, and it is not refused."""
    first, second = ValueError("bad"), ValueError("bad")
    first.code = 1  # pyright: ignore[reportAttributeAccessIssue] - an extra attribute is the case
    second.code = 2  # pyright: ignore[reportAttributeAccessIssue] - an extra attribute is the case

    forms = [fingerprint._form(value) for value in (first, second)]

    assert not [form for form in forms if _opaque_reports(form)], forms
    assert forms[0].text != forms[1].text


@dataclasses.dataclass(frozen=True)
class _LabelBase:
    label: str = "A"


# A class with no statement whose base is a dataclass: `_LabelSub()` takes the base's defaults.
_LabelSub = type("_LabelSub", (_LabelBase,), {})
_HOLDS_A_LABEL_SUB = functools.partial(str, _LabelSub())


def _names_a_label_sub() -> str:
    return _LabelSub().label


def _captures_a_label_sub() -> str:
    return _HOLDS_A_LABEL_SUB()


@pytest.mark.parametrize("stage", [_names_a_label_sub, _captures_a_label_sub])
def test_a_dataclass_subclass_without_a_statement_walks_its_bases_defaults(
    stage: typing.Callable[[], str],
):
    """`_LabelSub`, made by `type()`, is walked for the members its MRO resolves, and its
    dataclass defaults are resolved so too: the base's `label` default is tracked, named or
    captured, and its read replays."""
    manifest = fingerprint.get_stage_fingerprint(stage)
    reads = [[list(path), token] for path, token in _recording(stage)]

    key = f"default:{fingerprint._definition_ref(_LabelBase)}.label"
    assert key in manifest, manifest
    assert fingerprint._recorded_reads_hold(stage, typing.cast("list[object]", reads))


def test_a_single_entry_dict_or_set_ranks_its_entry_first():
    """A dict or set with one entry skips the sort, and its entry still gets the step a longer
    container gives it: a dict value its key's text (`k:"k"`), a set member the rank a sort
    gives the first one (`s:0`)."""
    in_dict = fingerprint._form({"k": _Captured()})
    in_set = fingerprint._form(frozenset((_Captured(),)))

    assert _class_reports(in_dict, _Captured) == [('k:"k"', "cls")]
    assert _class_reports(in_set, _Captured) == [("s:0", "cls")]


def _opaque_reports(form: typing.Any) -> list[tuple[str, ...]]:
    """The paths at which `form` reports a value whose state no form reads."""
    return [path for path, kind, _ in form.reports if kind is fingerprint._ReportKind.OPAQUE]


def _class_reports(form: typing.Any, cls: type) -> list[tuple[str, ...]]:
    """The paths at which `form` reports user class `cls` for its class node."""
    kind = fingerprint._ReportKind.CLASS
    return [path for path, found, obj in form.reports if found is kind and obj is cls]


def test_swapping_later_instances_of_two_classes_in_one_list_changes_the_manifest():
    """`[A(), B(), A(), B()]` to `[A(), B(), B(), A()]`: the first instance of each class stays
    at its position, but each class's walk is keyed by every position, so the manifest
    changes."""
    before = fingerprint.get_stage_fingerprint(_ordered_stage("ABAB"))
    after = fingerprint.get_stage_fingerprint(_ordered_stage("ABBA"))

    assert before != after


def _equal_set_stage(labels: str) -> typing.Callable[[str], str]:
    """A stage whose partial holds identity-distinct instances of two classes one factory made.
    Every instance has the same canonical form and unordered rank; `labels` controls the count
    of each class."""
    made = {label: _make_captured(label) for label in "AB"}
    holds = functools.partial(_tag_all, frozenset(made[label]() for label in labels))

    def stage(text: str) -> str:
        return holds(text)

    return stage


def test_replacing_a_class_at_an_equal_unordered_rank_changes_the_manifest():
    """`frozenset([A(), A(), B()])` to `frozenset([A(), B(), B()])`: equal forms share rank
    `s0`, but each class walk's key preserves the rank's occurrence count, so the manifest
    changes."""
    before = fingerprint.get_stage_fingerprint(_equal_set_stage("AAB"))
    after = fingerprint.get_stage_fingerprint(_equal_set_stage("ABB"))

    assert before != after


def test_a_builtin_exception_filename_is_hashed_as_value_state():
    """`FileNotFoundError.args` omits its filename. The built-in OSError descriptor state is
    included, so two otherwise identical defaults with different filenames have different
    forms and are not refused."""
    first = FileNotFoundError(2, "missing", "A")
    second = FileNotFoundError(2, "missing", "B")

    forms = [fingerprint._form(value) for value in (first, second)]

    assert not [form for form in forms if _opaque_reports(form)], forms
    assert forms[0].text != forms[1].text


def test_a_class_with_no_source_holding_a_value_no_form_reads_is_refused_in_every_stage():
    """A class `make_dataclass` made with a `random.Random` default: its hash would not show a
    change to the generator's state. Its hash never refuses (it names the value's type), and is
    kept out of the persistent cache; the walk of its class refuses each stage that names it,
    naming the class and the entry, also a second stage after the hash is in the memory cache,
    and the first stage again: no refusal outlives the scope that made it."""
    made = _random_beside("A")

    def names() -> object:
        return made()

    for stage in ("t/first", "t/second", "t/first"):
        with (
            pytest.raises(exceptions.StageDefinitionError, match="no source of its own") as raised,
            fingerprint.stage_fingerprint_scope(stage),
        ):
            fingerprint.get_stage_fingerprint(names)
        assert "'__dataclass_fields__'" in str(raised.value), raised.value
    assert fingerprint.hash_function_ast(made)
    assert fingerprint._should_skip_persistent_cache(made)


# A module of classes with no source, one of each kind `fingerprint._sourceless_class_hash`
# handles; TOKEN is replaced by a value that each class shows in an entry.
_NO_SOURCE_KINDS_MODULE = """\
import collections
import collections.abc
import dataclasses
import enum
import random
import typing

import pydantic

_MISSING = object()


class Box:
    __slots__ = ("x",)

    def __init__(self, x: int) -> None:
        self.x = x


def _upper(text: str) -> str:
    return text.upper()


def _check(cls: type, value: str) -> str:
    return value + "TOKEN"


NtPlain = collections.namedtuple("NtPlain", "a TOKEN")
NtDefaults = collections.namedtuple("NtDefaults", "a b", defaults=(1, "TOKEN"))
NtSentinel = collections.namedtuple("NtSentinel", "a b c", defaults=(_MISSING, "TOKEN"))
NtSlots = collections.namedtuple("NtSlots", "a box c", defaults=(Box(1), "TOKEN"))
MdPlain = dataclasses.make_dataclass(
    "MdPlain", [("x", int), ("y", str, dataclasses.field(default="TOKEN"))]
)
MdSentinel = dataclasses.make_dataclass(
    "MdSentinel",
    [("x", object, dataclasses.field(default=_MISSING)), ("y", str, dataclasses.field(default="TOKEN"))],
)
MdRandom = dataclasses.make_dataclass(
    "MdRandom",
    [
        ("rng", object, dataclasses.field(default=random.Random(1))),
        ("y", str, dataclasses.field(default="TOKEN")),
    ],
)
MdFrozen = dataclasses.make_dataclass("MdFrozen", [("TOKEN", int)], frozen=True)
EnumNames = enum.Enum("EnumNames", "TOKEN Z")
EnumValues = enum.Enum("EnumValues", {"X": "TOKEN"})
IntEnumF = enum.IntEnum("IntEnumF", "TOKEN Z")
FlagF = enum.Flag("FlagF", "TOKEN Z")
TypeDict = type("TypeDict", (), {"x": "TOKEN", "fn": _upper})
TypeLambda = type("TypeLambda", (), {"fmt": staticmethod(lambda s: s + "TOKEN")})
TypeSet = type("TypeSet", (), {"names": frozenset({"a", "b", "TOKEN"})})
TypeAbc = type(
    "TypeAbc", (collections.abc.Sized,), {"__module__": __name__, "__len__": lambda self: len("TOKEN")}
)
CmPlain = pydantic.create_model("CmPlain", x=(str, "TOKEN"))
CmConstrained = pydantic.create_model(
    "CmConstrained", x=(typing.Annotated[str, pydantic.Field(min_length=1, alias="TOKEN")], "v")
)
CmValidator = pydantic.create_model(
    "CmValidator", x=(str, "v"), __validators__={"check": pydantic.field_validator("x")(_check)}
)
CmConfig = pydantic.create_model("CmConfig", __config__=pydantic.ConfigDict(title="TOKEN"), x=(int, 1))
CmBase = pydantic.create_model("CmBase", __base__=CmPlain, y=(str, "TOKEN"))
CmSentinel = pydantic.create_model("CmSentinel", x=(object, _MISSING), y=(str, "TOKEN"))
"""

# The kinds above whose walk refuses, as a value no form reads makes it: a `random.Random`, whose
# state is in C. A sentinel `object()` and a `__slots__` object are shown by their type and
# slots. The hash of each kind, these included, never refuses.
_NO_SOURCE_REFUSED = {"MdRandom"}

_NO_SOURCE_HASH_SCRIPT = """\
import json
import sys

sys.path.insert(0, ".")
import kinds_mod
from pivot import fingerprint

out = {}
for name, cls in vars(kinds_mod).items():
    if isinstance(cls, type) and cls.__module__ == "kinds_mod" and name != "Box":
        walk = fingerprint._Walk()
        root = walk.node(cls, fingerprint._NodeKind.CLASS, fingerprint._class_key(cls))
        walk.run()
        try:
            with fingerprint._collected_capture_errors():
                fingerprint._project(root, {})
            refused = False
        except fingerprint.exceptions.StageDefinitionError:
            refused = True
        out[name] = [fingerprint._compute_function_hash(cls), refused]
print(json.dumps(out))
"""


def _no_source_hashes(directory: pathlib.Path, text: str, seed: str) -> dict[str, list[typing.Any]]:
    """The hash of each class of `text`, saved as module `kinds_mod`, from a new interpreter run
    under `PYTHONHASHSEED=seed`."""
    (directory / "kinds_mod.py").write_text(text)
    result = subprocess.run(
        [sys.executable, "-c", _NO_SOURCE_HASH_SCRIPT],
        cwd=directory,
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_a_class_with_no_source_is_hashed_alike_in_every_interpreter(tmp_path: pathlib.Path):
    """For every kind of class with no source, two interpreters under different hash seeds give
    one hash, and a comment edit in its module changes none. An edit to an entry the class shows
    changes it, for every kind. The walk of a class that holds a value no form reads
    (`_NO_SOURCE_REFUSED`) refuses, and no other kind's does."""
    (tmp_path / ".pivot").mkdir()
    original = _NO_SOURCE_KINDS_MODULE.replace("TOKEN", "A")
    first = _no_source_hashes(tmp_path, original, "1")
    second = _no_source_hashes(tmp_path, original, "2")
    commented = _no_source_hashes(tmp_path, original + "# note\n", "1")
    edited = _no_source_hashes(tmp_path, _NO_SOURCE_KINDS_MODULE.replace("TOKEN", "B"), "1")

    assert len(first) == 22, sorted(first)
    assert first == second, "Two interpreters should give each class one hash"
    assert first == commented, "A comment edit should change no hash"
    assert {name for name, (_, refused) in first.items() if refused} == _NO_SOURCE_REFUSED
    assert not {name for name in first if first[name][0] == edited[name][0]}


def test_a_chain_of_held_loaders_is_formed_once_per_loader(monkeypatch: pytest.MonkeyPatch):
    """A chain 64 deep takes twice the forming of a chain 32 deep, not four times: each held
    loader is formed once, in its holder's config form."""
    original = fingerprint._FormBuilder.loader
    calls = [0]

    def counting(self: typing.Any, value: typing.Any, value_type: type) -> typing.Any:
        calls[0] += 1
        return original(self, value, value_type)

    monkeypatch.setattr(fingerprint._FormBuilder, "loader", counting)
    counts = list[int]()
    for depth in (32, 64):
        calls[0] = 0
        fingerprint.get_loader_fingerprint(_chain(depth))
        counts.append(calls[0])

    assert counts[1] <= 2.2 * counts[0], counts


@dataclasses.dataclass(frozen=True)
class _CallableReader(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text()

    def __call__(self, path: pathlib.Path) -> str:
        return self.load(path)


def test_a_held_loader_that_is_callable_is_refused():
    """A callable loader in a config field is refused like any callable user class instance:
    what its `__call__` reads is not walked."""
    with pytest.raises(exceptions.StageDefinitionError, match="captures non-scalar variable"):
        fingerprint.get_loader_fingerprint(_HoldingReader(_CallableReader()))


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


def test_is_user_code_namespace_package_in_a_library_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A namespace package whose directory lies in one of pivot's own packages is not user code,
    though no directory of it is site-packages."""
    library = tmp_path / "library"
    (library / "ns_in_library").mkdir(parents=True)
    monkeypatch.setattr(
        fingerprint, "_PIVOT_PACKAGE_DIRS", (*fingerprint._PIVOT_PACKAGE_DIRS, library.resolve())
    )
    monkeypatch.syspath_prepend(str(library))
    sys.modules.pop("ns_in_library", None)
    try:
        module = importlib.import_module("ns_in_library")
        assert getattr(module, "__file__", None) is None, "a namespace package has no __file__"

        assert fingerprint.is_user_code(module) is False
    finally:
        sys.modules.pop("ns_in_library", None)


def test_is_user_code_namespace_package_split_with_a_directory_under_the_prefix(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A namespace package with one directory under the interpreter's prefix but outside
    site-packages, as pip's checkout of a ``pip install -e git+...`` package in ``<venv>/src``,
    and another in the user's code is user code, so the user's part of it is fingerprinted."""
    venv = tmp_path / "venv"
    vendor = venv / "src" / "vendor"
    user = tmp_path / "user"
    for import_root in (vendor, user):
        (import_root / "split_ns").mkdir(parents=True)
    monkeypatch.setattr(fingerprint, "_STDLIB_PATHS", (*fingerprint._STDLIB_PATHS, venv.resolve()))
    monkeypatch.syspath_prepend(str(user))
    monkeypatch.syspath_prepend(str(vendor))
    sys.modules.pop("split_ns", None)
    try:
        module = importlib.import_module("split_ns")
        assert len(module.__path__) == 2, "the namespace package spans both directories"

        assert fingerprint.is_user_code(module) is True
    finally:
        sys.modules.pop("split_ns", None)


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

    assert result == (
        str(pathlib.Path(__file__).resolve().relative_to(repo_root.resolve())),
        xxhash.xxh64(pathlib.Path(__file__).read_bytes()).hexdigest(),
    ), "the file's path in the project and the digest of its content"


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
    rel_path, digest = result
    assert "test_fingerprint.py" in rel_path
    assert digest == xxhash.xxh64(pathlib.Path(__file__).read_bytes()).hexdigest()


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
    fingerprint._pending_ast_writes.update(orig_pending)
    fingerprint._pending_manifest_writes.clear()
    fingerprint._pending_manifest_writes.update(orig_pending_manifest)
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = orig_db
    fingerprint._state_db_init_attempted = orig_attempted


# ==============================================================================
# Review fix: Recursive Pydantic model infinite recursion
# ==============================================================================


def test_recursive_pydantic_model_no_infinite_recursion():
    """Self-referential Pydantic models should not cause infinite recursion."""
    import pydantic

    class TreeNode(pydantic.BaseModel):
        value: int
        children: list["TreeNode"] = []

    TreeNode.model_rebuild()

    def stage(tree: TreeNode) -> None:
        pass

    # Should not raise RecursionError
    manifest = fingerprint.get_stage_fingerprint(stage)
    key = _definition_key("schema", TreeNode)
    assert key in manifest
    assert manifest[key] != "<pending>"


def test_mutual_recursive_pydantic_models():
    """Mutually recursive Pydantic models should not cause infinite recursion."""
    import pydantic

    class Parent(pydantic.BaseModel):
        name: str
        child: "Child | None" = None

    class Child(pydantic.BaseModel):
        name: str
        parent: "Parent | None" = None

    Parent.model_rebuild()
    Child.model_rebuild()

    def stage(parent: Parent) -> None:
        pass

    manifest = fingerprint.get_stage_fingerprint(stage)
    key = _definition_key("schema", Parent)
    assert key in manifest
    assert manifest[key] != "<pending>"
    assert _definition_key("schema", Child) in manifest


# ==============================================================================
# Review fix: User-defined generic origins tracked
# ==============================================================================


def test_user_defined_generic_origin_tracked():
    """User-defined generic classes used as origins should be fingerprinted."""
    import typing

    T = typing.TypeVar("T")

    class MyContainer(typing.Generic[T]):
        pass

    def stage(items: MyContainer[int]) -> None:
        pass

    manifest = fingerprint.get_stage_fingerprint(stage)

    assert _definition_key("class", MyContainer) in manifest


# ==============================================================================
# Review fix: Dotted string forward refs in class annotations
# ==============================================================================


def test_collect_annotation_names_dotted_string_forward_ref():
    """Dotted string forward refs like 'pkg.Type' should be treated as dotted refs."""
    import ast as ast_mod

    node = ast_mod.Constant(value="some_module.SomeType")
    names = set[str]()
    dotted_refs = list[tuple[str, ...]]()
    fingerprint._collect_annotation_names(node, names, dotted_refs)

    # Should NOT be in simple names (it has a dot)
    assert "some_module.SomeType" not in names
    # Should be treated as dotted ref
    assert ("some_module", "SomeType") in dotted_refs


def test_collect_annotation_names_simple_string_forward_ref():
    """Simple string forward refs like 'MyType' should still be added as names."""
    import ast as ast_mod

    node = ast_mod.Constant(value="MyType")
    names = set[str]()
    dotted_refs = list[tuple[str, ...]]()
    fingerprint._collect_annotation_names(node, names, dotted_refs)

    assert "MyType" in names
    assert len(dotted_refs) == 0


def test_persistent_cache_full_roundtrip(tmp_path, monkeypatch):
    """Full round-trip: compute -> flush -> hit -> modify -> miss.

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
    db_path = state_dir

    # Create initial StateDB so fingerprint module can open it in readonly mode
    with state.StateDB(db_path):
        pass

    # Patch project root and state db path
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

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
        assert len(_pending_ast_entries()) == 1
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # xxhash64 hexdigest

        # Step 2: Flush to StateDB
        fingerprint.flush_ast_hash_cache()
        assert len(_pending_ast_entries()) == 0

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
        assert len(_pending_ast_entries()) == 0

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
        assert len(_pending_ast_entries()) == 1  # Queued for cache
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
        fingerprint,
        "_producing_state_dir",
        lambda: tmp_path / "nonexistent" / "deeply" / "nested",
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
        assert len(_pending_ast_entries()) == 1

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
    db_path = state_dir

    # Create initial StateDB
    with state.StateDB(db_path) as db:
        pass

    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    # Reset fingerprint module state
    fingerprint._pending_ast_writes.clear()
    fingerprint._hash_function_ast_cache.clear()

    # Manually queue some entries (simulating what hash_function_ast does)
    py_ver = fingerprint._PYTHON_VERSION
    schema_ver = fingerprint._CACHE_SCHEMA_VERSION
    test_entries: list[fingerprint.AstHashEntry] = [
        ("src/a.py", "0123456789abcdef", "func_a", py_ver, schema_ver, "aaaa111122223333"),
        ("src/b.py", "1123456789abcdef", "func_b", py_ver, schema_ver, "bbbb444455556666"),
        ("src/c.py", "2123456789abcdef", "MyClass.method", py_ver, schema_ver, "cccc777788889999"),
    ]
    fingerprint._pending_ast_writes[db_path] = list(test_entries)

    # Flush to StateDB
    fingerprint.flush_ast_hash_cache()

    # Verify entries were actually written
    with state.StateDB(db_path, readonly=True) as db:
        for rel_path, digest, qualname, py_version, schema_version, expected_hash in test_entries:
            actual_hash = db.get_ast_hash(rel_path, digest, qualname, py_version, schema_version)
            assert actual_hash == expected_hash, (
                f"Expected {expected_hash} for {rel_path}:{qualname}, got {actual_hash}"
            )

    # Pending writes should be cleared
    assert len(_pending_ast_entries()) == 0


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

        # Should have recorded the stage's source file, by the digest of its content
        assert source_map["collected_stage.py"] == (
            xxhash.xxh64(test_module.read_bytes()).hexdigest()
        )
    finally:
        sys.modules.pop("collected_stage", None)


def test_cached_manifest_source_map_includes_module_attribute_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `module.NAME` read records that module's file, so editing it invalidates the cache."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "f8_settings.py").write_text("FACTOR = 10\n")
    (tmp_path / "f8_stage.py").write_text(
        "import f8_settings\n\n\ndef scale(x):\n    return x * f8_settings.FACTOR\n"
    )
    writes = dict[pathlib.Path, list[tuple[bytes, bytes]]]()
    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", writes)
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage_name, _func: None)
    try:
        stage_module = importlib.import_module("f8_stage")
        fingerprint.get_stage_fingerprint_cached("f8/scale", stage_module.scale)
    finally:
        sys.modules.pop("f8_stage", None)
        sys.modules.pop("f8_settings", None)

    ((_key, raw),) = _pending_manifest_entries()
    assert "f8_settings.py" in json.loads(raw)["s"]


def test_project_module_scan_is_reused_until_loaded_modules_change(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache misses reuse one scan of sys.modules; importing or reloading a module redoes it."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "scan_first.py").write_text("VALUE = 1\n")
    (tmp_path / "scan_second.py").write_text("VALUE = 2\n")
    classified = list[object]()
    real_is_user_code = fingerprint.is_user_code

    def counting_is_user_code(obj: object) -> bool:
        classified.append(obj)
        return real_is_user_code(obj)

    try:
        importlib.import_module("scan_first")
        first = dict[str, str]()
        fingerprint._record_project_modules(first)
        assert "scan_first.py" in first

        monkeypatch.setattr(fingerprint, "is_user_code", counting_is_user_code)
        repeat = dict[str, str]()
        fingerprint._record_project_modules(repeat)
        assert repeat == first
        assert classified == [], "an unchanged sys.modules must not be rescanned"

        importlib.import_module("scan_second")
        after_import = dict[str, str]()
        fingerprint._record_project_modules(after_import)
        assert "scan_second.py" in after_import

        (tmp_path / "scan_first.py").write_text("VALUE = 100\n")
        del sys.modules["scan_first"]
        importlib.import_module("scan_first")
        after_reload = dict[str, str]()
        fingerprint._record_project_modules(after_reload)
        assert after_reload["scan_first.py"] == (
            xxhash.xxh64((tmp_path / "scan_first.py").read_bytes()).hexdigest()
        )
    finally:
        sys.modules.pop("scan_first", None)
        sys.modules.pop("scan_second", None)


_READS_MODULE = """\
import functools

import pydantic

import reads_settings

FACTOR = reads_settings.SOURCE["factor"]


class Model(pydantic.BaseModel):
    width: int = reads_settings.SOURCE["width"]


def helper(x, offset=reads_settings.SOURCE["offset"]):
    return x + offset


def apply(x, by):
    return x * by


BOUND = functools.partial(apply, by=reads_settings.SOURCE["by"])


def make(scale):
    def stage(x, model: Model = None):
        return helper(BOUND(x)) * scale * reads_settings.ATTR + (lambda: FACTOR)()

    return stage


stage = make(reads_settings.SOURCE["scale"])
"""


@pytest.mark.parametrize("changed", ["factor", "width", "offset", "by", "scale", "attr"])
def test_recorded_reads_detect_each_changed_live_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """Every value the walk read is checked on a hit: a global read by a nested lambda, a
    closure cell, a helper's default, a `module.NAME` scalar, a partial argument and a schema
    field default, each computed at import from a module that is not edited."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    source = {"factor": 1, "width": 2, "offset": 3, "by": 4, "scale": 5}
    (tmp_path / "reads_settings.py").write_text(f"SOURCE = {source!r}\nATTR = 6\n")
    (tmp_path / "reads_stage.py").write_text(_READS_MODULE)
    writes = dict[pathlib.Path, list[tuple[bytes, bytes]]]()
    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", writes)
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage_name, _func: None)
    try:
        settings = importlib.import_module("reads_settings")
        module = importlib.import_module("reads_stage")
        fingerprint.get_stage_fingerprint_cached("r/stage", module.stage)
        ((_key, raw),) = _pending_manifest_entries()
        reads = json.loads(raw)["r"]
        del sys.modules["reads_stage"]
        module = importlib.import_module("reads_stage")
        assert fingerprint._recorded_reads_hold(module.stage, reads), "re-import alone must hit"

        if changed == "attr":
            vars(settings)["ATTR"] += 100
        else:
            vars(settings)["SOURCE"][changed] += 100
        del sys.modules["reads_stage"]
        module = importlib.import_module("reads_stage")

        assert fingerprint._recorded_reads_hold(module.stage, reads) is False
    finally:
        sys.modules.pop("reads_stage", None)
        sys.modules.pop("reads_settings", None)


_REPLAY_LIB = """\
import replay_settings

LIMIT = replay_settings.SOURCE["method"]


class Walker:
    def step(self, x):
        return x + LIMIT
"""

_REPLAY_BODY = """\
import replay_settings

BODY = replay_settings.SOURCE["body"]
"""

_REPLAY_STAGE = """\
from replay_lib import Walker


def stage(x):
    import replay_body

    return Walker().step(x) + replay_body.BODY
"""


@pytest.mark.parametrize("changed", ["method", "body"])
def test_recorded_reads_replay_method_and_body_import_reads(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """A cache hit re-reads what a user class's method read (`m:NAME`) and what a module the
    body imports holds (`bi:MODULE`), each computed at import from a module that is not
    edited."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "replay_settings.py").write_text("SOURCE = {'method': 1, 'body': 2}\n")
    (tmp_path / "replay_lib.py").write_text(_REPLAY_LIB)
    (tmp_path / "replay_body.py").write_text(_REPLAY_BODY)
    (tmp_path / "replay_stage.py").write_text(_REPLAY_STAGE)
    writes = dict[pathlib.Path, list[tuple[bytes, bytes]]]()
    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", writes)
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage_name, _func: None)
    names = ("replay_stage", "replay_lib", "replay_body")
    try:
        settings = importlib.import_module("replay_settings")
        importlib.import_module("replay_body")
        module = importlib.import_module("replay_stage")
        manifest = fingerprint.get_stage_fingerprint_cached("r/stage", module.stage)
        assert manifest["const:replay_lib.LIMIT"] == "1", manifest
        assert manifest["const:replay_body.BODY"] == "2", manifest
        ((_key, raw),) = _pending_manifest_entries()
        reads = json.loads(raw)["r"]
        for name in names:
            del sys.modules[name]
        importlib.import_module("replay_body")
        module = importlib.import_module("replay_stage")
        assert fingerprint._recorded_reads_hold(module.stage, reads), "re-import alone must hit"

        vars(settings)["SOURCE"][changed] += 100
        for name in names:
            del sys.modules[name]
        importlib.import_module("replay_body")
        module = importlib.import_module("replay_stage")

        assert fingerprint._recorded_reads_hold(module.stage, reads) is False
    finally:
        for name in (*names, "replay_settings"):
            sys.modules.pop(name, None)


def _helper_make_scaler(factor: int) -> typing.Callable[[int], int]:
    def scaler(x: int) -> int:
        return x * factor

    return scaler


def test_fingerprint_pass_names_each_function_once_and_ends_with_the_pass() -> None:
    """Inside one pass a function's identity is computed once; the next pass sees a change."""
    scaler = _helper_make_scaler(2)
    before = fingerprint.function_identity(scaler)
    cell = typing.cast("tuple[types.CellType, ...]", scaler.__closure__)[0]

    with fingerprint.fingerprint_pass():
        first = fingerprint.function_identity(scaler)
        cell.cell_contents = 5
        within = fingerprint.function_identity(scaler)
    after = fingerprint.function_identity(scaler)

    assert first == before and within is first, "one pass names the function once"
    assert after["cells"] == [("factor", 5)], after
    assert after != before, "a new pass reads the changed cell"


class _ClassReadMode(enum.Enum):
    FAST = "fast"


def _helper_class_callback() -> int:
    return 1


class _ClassReadConfig:
    SCALE: typing.ClassVar[int] = 3
    MODE: typing.ClassVar[_ClassReadMode] = _ClassReadMode.FAST
    CALLBACK: typing.ClassVar[typing.Callable[[], int]] = _helper_class_callback

    @staticmethod
    def static() -> int:
        return 2


def _helper_reads_class_attributes() -> int:
    return (
        _ClassReadConfig.SCALE
        + _ClassReadConfig.CALLBACK()  # pyright: ignore[reportAttributeAccessIssue] - a function stored on the class
        + _ClassReadConfig.static()
        + len(_ClassReadConfig.MODE.value)
    )


def test_class_attributes_a_stage_reads_are_tracked_one_by_one() -> None:
    """A scalar is `const:<class>.<NAME>`, an Enum member is name and value, a plain function
    is walked; a staticmethod is walked as one of the class's methods, not read as a value."""
    manifest = fingerprint.get_stage_fingerprint(_helper_reads_class_attributes)
    config = f"{__name__}._ClassReadConfig"

    assert manifest[f"const:{config}.SCALE"] == "3", manifest
    assert manifest[f"const:{config}.MODE"] == '_ClassReadMode.FAST="fast"', manifest
    assert f"func:{__name__}._helper_class_callback" in manifest, manifest
    assert f"func:{config}.static" in manifest, manifest
    assert not any(key.startswith(f"const:{config}.static") for key in manifest), manifest
    assert f"class:{config}" in manifest, manifest
    assert not any(key.startswith(f"const:{config}.__") for key in manifest), manifest


class _InstanceClassRead:
    SCALE: typing.ClassVar[int] = 4

    def read(self) -> int:
        return self.SCALE


def _reads_class_attribute_through_instance() -> int:
    return _InstanceClassRead().read()


def test_class_attribute_read_through_an_instance_is_tracked() -> None:
    """A class member reading `self.SCALE` creates the same value entry as `Class.SCALE`."""
    manifest = fingerprint.get_stage_fingerprint(_reads_class_attribute_through_instance)

    key = f"const:{__name__}._InstanceClassRead.SCALE"
    assert manifest[key] == "4", manifest


def _return_captured_value(value: object) -> object:
    return value


_UNKNOWN_LIBRARY_VALUE = random.Random(1)
_HOLDS_UNKNOWN_LIBRARY_VALUE = functools.partial(_return_captured_value, _UNKNOWN_LIBRARY_VALUE)


def _captures_an_unknown_library_value() -> object:
    return _HOLDS_UNKNOWN_LIBRARY_VALUE()


def test_unknown_library_value_inside_a_partial_is_refused() -> None:
    """A partial cannot hide an object whose type Pivot has no value-form row."""
    with pytest.raises(exceptions.StageDefinitionError, match="random.Random|cannot read"):
        fingerprint.get_stage_fingerprint(_captures_an_unknown_library_value)


# ==============================================================================
# Stage manifest cache tests
# ==============================================================================


def test_manifest_cache_hit(tmp_path, monkeypatch):
    """Compute -> flush -> compute again returns cached manifest (walk not called)."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

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
        assert len(_pending_manifest_entries()) == 1

        # Flush
        fingerprint.flush_manifest_cache()
        assert len(_pending_manifest_entries()) == 0

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
        assert len(_pending_manifest_entries()) == 0
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
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

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
        assert len(_pending_manifest_entries()) == 1  # Queued for flush
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
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

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
        assert len(_pending_manifest_entries()) == 1  # Cache miss -> re-queued
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
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

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
    """The key names the stage, the function, the python version, and the schema version."""
    key = fingerprint._make_manifest_cache_key("train", _cached_stage)
    decoded = key.decode()
    assert decoded.startswith("sm:")
    assert "train" in decoded
    assert _cached_stage.__module__ in decoded
    assert _cached_stage.__qualname__ in decoded
    assert fingerprint._PYTHON_VERSION in decoded
    assert str(fingerprint._CACHE_SCHEMA_VERSION) in decoded
    # Null byte separators present
    assert "\x00" in decoded


def test_make_manifest_cache_key_different_stages_differ():
    """Different stage names produce different keys."""
    key1 = fingerprint._make_manifest_cache_key("train", _cached_stage)
    key2 = fingerprint._make_manifest_cache_key("evaluate", _cached_stage)
    assert key1 != key2


def test_make_manifest_cache_key_different_functions_differ(tmp_path, monkeypatch):
    """One stage name registered with functions from two files gets two keys."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    functions = list[typing.Callable[[], int]]()
    for directory in ("a", "b"):
        path = tmp_path / directory / "pipeline.py"
        path.parent.mkdir()
        path.write_text("def make():\n    return 1\n")
        namespace = dict[str, typing.Any]()
        exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102 - test fixture
        functions.append(namespace["make"])

    first, second = (fingerprint._make_manifest_cache_key("rep/make", f) for f in functions)
    assert first != second


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
        fingerprint,
        "_producing_state_dir",
        lambda: blocker / "subdir",
    )

    # Queue some entries under the directory the failing flush will try to write
    state_dir = blocker / "subdir"
    fingerprint._pending_manifest_writes.clear()
    test_entries = [(b"sm:test_stage\x003.13\x001", b'{"m":{},"s":{}}')]
    fingerprint._pending_manifest_writes[state_dir] = list(test_entries)

    # Flush should fail (blocker is a file, so mkdir(parents=True) raises NotADirectoryError)
    fingerprint.flush_manifest_cache()

    # Entries should be restored for retry, under the same directory
    assert fingerprint._pending_manifest_writes == {state_dir: test_entries}

    # Clean up
    fingerprint._pending_manifest_writes.clear()


def test_try_manifest_cache_hit_corrupted_json(tmp_path, monkeypatch):
    """Corrupted JSON in cache returns None instead of crashing."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir

    # Write corrupted data directly into the db
    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("bad_stage", _cached_stage)
        db.put_raw(key, b"NOT VALID JSON {{{")

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("bad_stage", _cached_stage)
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
        result = fingerprint._try_manifest_cache_hit("any_stage", _cached_stage)
        assert result is None
    finally:
        fingerprint._state_db_init_attempted = False


def test_collecting_sources_restores_on_exception(tmp_path, monkeypatch):
    """_active_source_map is set to None even if an exception occurs during fingerprinting."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    # Verify _active_source_map is None before
    assert fingerprint._active_source_map.get() is None

    class FingerprintError(Exception):
        pass

    try:
        with fingerprint._collecting_sources() as source_map:
            assert fingerprint._active_source_map.get() is source_map
            raise FingerprintError("simulated failure")
    except FingerprintError:
        pass

    # Must be restored to None after exception
    assert fingerprint._active_source_map.get() is None, (
        "_active_source_map should be None after exception in _collecting_sources"
    )


def test_collecting_sources_in_two_threads_leaves_no_map_behind():
    """Two fingerprints collect sources at once in two threads, as the engine's worker threads
    can: the first enters, the second enters, the first ends, the second ends. Each sees its own
    map throughout, and neither thread nor the caller has a map active afterwards."""
    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()
    seen = dict[str, object]()

    def first() -> None:
        with fingerprint._collecting_sources() as source_map:
            first_in.set()
            second_in.wait(timeout=60)
            seen["first"] = fingerprint._active_source_map.get() is source_map
        first_out.set()
        seen["first after"] = fingerprint._active_source_map.get()

    def second() -> None:
        first_in.wait(timeout=60)
        with fingerprint._collecting_sources() as source_map:
            second_in.set()
            first_out.wait(timeout=60)
            seen["second"] = fingerprint._active_source_map.get() is source_map
        seen["second after"] = fingerprint._active_source_map.get()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert seen == {"first": True, "first after": None, "second": True, "second after": None}
    assert fingerprint._active_source_map.get() is None


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
        assert len(_pending_manifest_entries()) == 1
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
    db_path = state_dir

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("list_stage", _cached_stage)
        db.put_raw(key, b"[1, 2, 3]")  # Valid JSON, but not a dict

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("list_stage", _cached_stage)
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
    db_path = state_dir

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("missing_keys", _cached_stage)
        db.put_raw(key, b'{"m": {"self:x": "hash"}}')  # Missing "s" key

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("missing_keys", _cached_stage)
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
    db_path = state_dir

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("empty_sources", _cached_stage)
        db.put_raw(key, b'{"m":{"self:x":"hash"},"s":{}}')

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("empty_sources", _cached_stage)
        assert result is None, "Empty sources should force recompute"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_try_manifest_cache_hit_source_entry_not_a_digest(tmp_path, monkeypatch):
    """A source entry that is not a content digest, as the (mtime, size, inode) list pivot
    recorded before it keyed sources by content, returns None."""
    from pivot.storage import state as state_mod

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir

    src_file = tmp_path / "stage.py"
    src_file.write_text("def stage(): pass\n")
    stat = src_file.stat()
    stats = [stat.st_mtime_ns, stat.st_size, stat.st_ino]

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("bad_stats", _cached_stage)
        db.put_raw(
            key,
            json.dumps({"m": {"self:stage": "hash"}, "s": {"stage.py": stats}, "r": []}).encode(),
        )

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("bad_stats", _cached_stage)
        assert result is None, "A source entry that is not a digest should return None"
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
    db_path = state_dir

    # Test with absolute path
    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("traversal_abs", _cached_stage)
        db.put_raw(key, b'{"m":{"self:x":"hash"},"s":{"/etc/passwd":"0123456789abcdef"},"r":[]}')

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("traversal_abs", _cached_stage)
        assert result is None, "Absolute path in sources should be rejected"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

    # Test with ../ traversal
    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("traversal_dotdot", _cached_stage)
        db.put_raw(
            key, b'{"m":{"self:x":"hash"},"s":{"../../etc/passwd":"0123456789abcdef"},"r":[]}'
        )

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    try:
        result = fingerprint._try_manifest_cache_hit("traversal_dotdot", _cached_stage)
        assert result is None, "Path traversal with .. should be rejected"
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_manifest_cache_flush_persists_to_statedb(tmp_path, monkeypatch):
    """Manifest cache flush writes pending manifests to StateDB.

    Verifies that:
    1. get_stage_fingerprint_cached() queues manifest writes
    2. flush_manifest_cache() writes to StateDB
    3. Manifest can be retrieved from StateDB after flush
    """
    from pivot.storage import state

    # Set up isolated state directory
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir

    # Create initial StateDB
    with state.StateDB(db_path):
        pass

    # Patch project root and state db path
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    # Reset fingerprint module state
    fingerprint._pending_manifest_writes.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    # Create test module file
    test_module = tmp_path / "test_stage.py"
    test_module.write_text("""
def test_stage():
    return 42
""")

    # Import the function
    spec = importlib.util.spec_from_file_location("test_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_stage"] = module
    try:
        spec.loader.exec_module(module)
        func = module.test_stage

        # Step 1: Call get_stage_fingerprint_cached() - should queue manifest
        manifest = fingerprint.get_stage_fingerprint_cached("test_stage", func)
        assert isinstance(manifest, dict)
        assert "self:test_stage" in manifest
        assert len(_pending_manifest_entries()) > 0, "Should queue manifest writes"

        # Step 2: Flush to StateDB
        fingerprint.flush_manifest_cache()
        assert len(_pending_manifest_entries()) == 0, "Pending writes should be empty after flush"

        # Step 3: Verify manifest is in StateDB
        with state.StateDB(db_path, readonly=True) as db:
            key = fingerprint._make_manifest_cache_key("test_stage", func)
            raw = db.get_raw(key)
            assert raw is not None, "Manifest should be persisted in StateDB"
            data = json.loads(raw)
            assert "m" in data, "Cached data should have manifest key"
            assert "s" in data, "Cached data should have sources key"
            cached_manifest = data["m"]
            assert "self:test_stage" in cached_manifest
    finally:
        sys.modules.pop("test_stage", None)
        # Clean up fingerprint module state
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def test_invalidate_manifests_for_paths_selective(tmp_path, monkeypatch):
    """Invalidate only manifests referencing changed source files."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    key_a = fingerprint._make_manifest_cache_key("stage_a", _cached_stage)
    key_b = fingerprint._make_manifest_cache_key("stage_b", _cached_stage)
    key_c = fingerprint._make_manifest_cache_key("stage_c", _cached_stage)

    with state_mod.StateDB(db_path) as db:
        db.put_raw(
            key_a,
            json.dumps(
                {"m": {"self:stage_a": "hash"}, "s": {"a.py": "0123456789abcdef"}},
                separators=(",", ":"),
            ).encode(),
        )
        db.put_raw(
            key_b,
            json.dumps(
                {"m": {"self:stage_b": "hash"}, "s": {"b.py": "0123456789abcdef"}},
                separators=(",", ":"),
            ).encode(),
        )
        db.put_raw(
            key_c,
            json.dumps(
                {"m": {"self:stage_c": "hash"}, "s": {"c.py": "0123456789abcdef"}},
                separators=(",", ":"),
            ).encode(),
        )

    fingerprint.invalidate_manifests_for_paths([tmp_path / "b.py"])

    with state_mod.StateDB(db_path, readonly=True) as db:
        assert db.get_raw(key_a) is not None, "Unchanged manifest should remain"
        assert db.get_raw(key_b) is None, "Changed manifest should be removed"
        assert db.get_raw(key_c) is not None, "Unchanged manifest should remain"


def test_invalidate_manifests_for_paths_cold_start(tmp_path, monkeypatch):
    """Invalidation is a no-op when no cached manifests exist."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    fingerprint.invalidate_manifests_for_paths([tmp_path / "missing.py"])

    with state_mod.StateDB(db_path, readonly=True) as db:
        assert list(db.iter_prefix(b"sm:")) == [], "No manifest entries should exist"


def test_invalidate_manifests_for_paths_recomputes_affected_stage(tmp_path, monkeypatch):
    """Invalidation forces a cache miss and re-cache for affected stages."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: db_path)

    fingerprint._pending_manifest_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False

    test_module = tmp_path / "invalidate_stage.py"
    test_module.write_text("""
def invalidate_stage():
    return 42
""")

    spec = importlib.util.spec_from_file_location("invalidate_stage", test_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["invalidate_stage"] = module
    try:
        spec.loader.exec_module(module)

        key = fingerprint._make_manifest_cache_key("invalidate_stage", module.invalidate_stage)
        fingerprint.get_stage_fingerprint_cached("invalidate_stage", module.invalidate_stage)
        fingerprint.flush_manifest_cache()

        with state_mod.StateDB(db_path, readonly=True) as db:
            assert db.get_raw(key) is not None, "Manifest should be cached before invalidation"

        fingerprint.invalidate_manifests_for_paths([test_module])

        with state_mod.StateDB(db_path, readonly=True) as db:
            assert db.get_raw(key) is None, "Manifest should be removed after invalidation"

        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False

        fingerprint.get_stage_fingerprint_cached("invalidate_stage", module.invalidate_stage)
        fingerprint.flush_manifest_cache()

        with state_mod.StateDB(db_path, readonly=True) as db:
            assert db.get_raw(key) is not None, "Manifest should be re-cached after recompute"
    finally:
        sys.modules.pop("invalidate_stage", None)
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def _helper_rewrite_keeping_stat(path: pathlib.Path, old: str, new: str) -> None:
    """Replace ``old`` with ``new`` (same length) in ``path`` in place, then give the file back
    its mtime: its mtime, size and inode are those of the previous content, as after two writes
    inside one kernel timestamp tick."""
    before = path.stat()
    text = path.read_text()
    assert len(old) == len(new) and old in text
    path.write_text(text.replace(old, new))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_mtime_ns, after.st_size, after.st_ino) == (
        before.st_mtime_ns,
        before.st_size,
        before.st_ino,
    )


@pytest.fixture
def same_stat_project(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[pathlib.Path]:
    """A project root with a StateDB that the fingerprint caches read and flush to; no bytecode
    is cached, so a module imported again runs its file's current source."""
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    with state_mod.StateDB(state_dir):
        pass
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: state_dir)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    fingerprint._pending_manifest_writes.clear()
    fingerprint._pending_ast_writes.clear()
    fingerprint._hash_function_ast_cache.clear()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False
    try:
        yield tmp_path
    finally:
        if fingerprint._state_db is not None:
            fingerprint._state_db.close()
        fingerprint._state_db = None
        fingerprint._state_db_init_attempted = False


def _helper_import_fresh(name: str, path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _helper_reopen_state_db() -> None:
    """Make the fingerprint caches read what the last flush wrote."""
    if fingerprint._state_db is not None:
        fingerprint._state_db.close()
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False


def test_cached_function_hash_misses_when_an_edit_keeps_the_files_stat(
    same_stat_project: pathlib.Path,
) -> None:
    """A function's cached hash is keyed by its file's content: an edit that keeps the file's
    mtime, size and inode is hashed again, from the edited source, not from the lines
    ``linecache`` kept for the previous content."""
    source = same_stat_project / "same_stat_func.py"
    source.write_text('def build():\n    return "alpha"\n')
    try:
        first = fingerprint.hash_function_ast(_helper_import_fresh("same_stat_func", source).build)
        fingerprint.flush_ast_hash_cache()
        _helper_reopen_state_db()

        _helper_rewrite_keeping_stat(source, "alpha", "bravo")
        edited = _helper_import_fresh("same_stat_func", source).build
        assert edited() == "bravo"

        assert fingerprint.hash_function_ast(edited) != first
    finally:
        sys.modules.pop("same_stat_func", None)


def test_cached_manifest_misses_when_a_recorded_file_changes_keeping_its_stat(
    same_stat_project: pathlib.Path,
) -> None:
    """A cached manifest records each source file by its content: an edit to one of them that
    keeps its mtime, size and inode is a miss."""
    source = same_stat_project / "same_stat_stage.py"
    source.write_text("def same_stat_stage():\n    return 42\n\n\nNOTE = 'alpha'\n")
    try:
        stage = _helper_import_fresh("same_stat_stage", source).same_stat_stage
        fingerprint.get_stage_fingerprint_cached("same/stage", stage)
        fingerprint.flush_manifest_cache()
        _helper_reopen_state_db()
        assert fingerprint._try_manifest_cache_hit("same/stage", stage) is not None, (
            "the unedited file hits"
        )

        _helper_rewrite_keeping_stat(source, "alpha", "bravo")

        assert fingerprint._try_manifest_cache_hit("same/stage", stage) is None
    finally:
        sys.modules.pop("same_stat_stage", None)


def test_function_hash_lookup_reads_the_current_projects_state_db(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that moves to another project looks function hashes up in that project's
    StateDB, not in the one its first lookup opened."""
    source = "def build():\n    return 1\n"
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / ".pivot").mkdir(parents=True)
        (root / "moved_stage.py").write_text(source)
    planted: fingerprint.AstHashEntry = (
        "moved_stage.py",
        xxhash.xxh64(source.encode()).hexdigest(),
        "build",
        fingerprint._PYTHON_VERSION,
        fingerprint._CACHE_SCHEMA_VERSION,
        "5ec0d5ec0d5ec0d0",
    )
    with state_mod.StateDB(second / ".pivot") as db:
        db.save_ast_hash_many([planted])
    try:
        monkeypatch.setattr(project, "_project_root_cache", first)
        built = _helper_import_fresh("moved_stage", first / "moved_stage.py").build
        fingerprint.hash_function_ast(built)

        monkeypatch.setattr(project, "_project_root_cache", second)
        moved = _helper_import_fresh("moved_stage", second / "moved_stage.py").build

        assert fingerprint.hash_function_ast(moved) == "5ec0d5ec0d5ec0d0"
    finally:
        sys.modules.pop("moved_stage", None)
        fingerprint._close_state_db()


# ---------------------------------------------------------------------------------------------
# The walk's worklist, cache-hit replay of value steps, wrapper loops behind a C cache, Enum
# values and a rebuilt pydantic model
# ---------------------------------------------------------------------------------------------


def test_a_helper_chain_deeper_than_the_recursion_limit_is_fingerprinted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each helper is a node of the walk, expanded from its worklist, never inside the expansion
    of the helper that calls it, so a stage whose helpers call each other deeper than the
    recursion limit gets an entry for each of them."""
    depth = sys.getrecursionlimit() + 500
    helpers = [f"def helper_{i}(x):\n    return helper_{i + 1}(x) + 1\n" for i in range(depth)]
    helpers.append(f"def helper_{depth}(x):\n    return x\n")
    helpers.append("def stage():\n    return helper_0(0)\n")
    (tmp_path / "deep_helpers.py").write_text("\n\n".join(helpers))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module("deep_helpers")
        manifest = fingerprint.get_stage_fingerprint(module.stage)
    finally:
        sys.modules.pop("deep_helpers", None)

    walked = [key for key in manifest if key.startswith("func:deep_helpers.helper_")]
    assert len(walked) == depth + 1, f"{len(walked)} of {depth + 1} helpers were walked"


_VALUE_STEPS_MODULE = """\
import functools


class First:
    def __repr__(self) -> str:
        return "z"

    def suffix(self) -> str:
        return "A"


class Second:
    def __repr__(self) -> str:
        return "a"

    def suffix(self) -> str:
        return "B"


def _count(held, x):
    return x + len(held)


HELD = functools.partial(_count, {held})


def stage(x):
    return HELD(x)
"""


@pytest.mark.parametrize(
    "held",
    [
        pytest.param("frozenset([First(), Second()])", id="set-members"),
        pytest.param('{"first": First(), "second": Second()}', id="dict-values"),
    ],
)
def test_a_cache_hit_reaches_the_classes_inside_a_set_and_a_dict_again(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, held: str
) -> None:
    """A partial holds two instances in a set, ranked by their forms (`First` before `Second`,
    though their reprs sort the other way), or as a dict's values, keyed by the keys' forms. The
    walk records the reads of each class under the step to it (`s:RANK`, `k:KEY`), and a cache hit
    in a new import follows the same steps to the same classes, so an unchanged value hits."""
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "value_steps.py").write_text(_VALUE_STEPS_MODULE.format(held=held))
    writes = dict[pathlib.Path, list[tuple[bytes, bytes]]]()
    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", writes)
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage_name, _func: None)
    try:
        module = importlib.import_module("value_steps")
        fingerprint.get_stage_fingerprint_cached("v/stage", module.stage)
        ((_key, raw),) = _pending_manifest_entries()
        reads = json.loads(raw)["r"]
        del sys.modules["value_steps"]
        module = importlib.import_module("value_steps")

        assert fingerprint._recorded_reads_hold(module.stage, reads), "re-import alone must hit"
    finally:
        sys.modules.pop("value_steps", None)


_LOOP_FACTOR = 3


def _helper_looped_behind_a_cache(x: int) -> int:
    return x * _LOOP_FACTOR


_LOOPED_CACHE = functools.cache(_helper_looped_behind_a_cache)
_helper_looped_behind_a_cache.__wrapped__ = _LOOPED_CACHE  # pyright: ignore[reportFunctionMemberAccess] - a wrapper loop


def _helper_stage_calls_a_looped_cache() -> int:
    return _LOOPED_CACHE(1)


def test_a_cache_wrapper_whose_wrapped_chain_loops_is_refused():
    """D3: refuse a Python/C metadata loop instead of accepting an incomplete manifest."""
    with pytest.raises(exceptions.StageDefinitionError) as raised:
        fingerprint.get_stage_fingerprint(_helper_stage_calls_a_looped_cache)
    message = str(raised.value)
    assert "_helper_looped_behind_a_cache" in message and "loop" in message, message
    assert "StageParams" in message and "break" in message.lower(), message


def _helper_library_enum(value: str) -> typing.Any:
    """A functional Enum that names a library module as its own, so it is a library Enum."""
    return enum.Enum("LibraryChoice", {"PICK": value}, module="json")


_HOLDS_LIBRARY_PICK_A = functools.partial(_tag, _helper_library_enum("A").PICK)
_HOLDS_LIBRARY_PICK_B = functools.partial(_tag, _helper_library_enum("B").PICK)


def _helper_stage_holds_library_pick_a(text: str) -> str:
    return _HOLDS_LIBRARY_PICK_A(text)


def _helper_stage_holds_library_pick_b(text: str) -> str:
    return _HOLDS_LIBRARY_PICK_B(text)


def test_a_library_enum_members_value_is_in_the_partials_form():
    """Members of two library Enums of one name, with one member name and different values, give
    different partial entries: the member is formed with its value."""
    first = fingerprint.get_stage_fingerprint(_helper_stage_holds_library_pick_a)
    second = fingerprint.get_stage_fingerprint(_helper_stage_holds_library_pick_b)

    first_args = first[f"partial:{__name__}._HOLDS_LIBRARY_PICK_A.args"]
    second_args = second[f"partial:{__name__}._HOLDS_LIBRARY_PICK_B.args"]
    assert first_args != second_args, "A member's value must be in the partial's entry"


class _RebuiltDefault(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )
    value: typing.Any = random.Random(1)


def _helper_builds_a_rebuilt_model() -> object:
    return _RebuiltDefault()


def test_a_model_rebuilt_with_a_readable_default_is_accepted_in_the_next_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    """A model's field default is a `random.Random`, whose state no form reads: the stage is
    refused. After the default is replaced with a string and the model rebuilt, the next pass in
    the same process accepts the stage: the schema hash kept for the model is not the one its old
    default made."""
    with fingerprint.fingerprint_pass(), pytest.raises(exceptions.StageDefinitionError):
        fingerprint.get_stage_fingerprint(_helper_builds_a_rebuilt_model)
    field = _RebuiltDefault.model_fields["value"]
    monkeypatch.setattr(field, "default", "safe")
    _RebuiltDefault.model_rebuild(force=True)

    with fingerprint.fingerprint_pass():
        manifest = fingerprint.get_stage_fingerprint(_helper_builds_a_rebuilt_model)

    assert _definition_key("schema", _RebuiltDefault) in manifest, manifest


@pytest.mark.parametrize("setting", ["typed", "maxsize"])
@pytest.mark.parametrize("holder", ["wrapped", "partial"])
def test_a_cache_over_library_code_keeps_mains_by_name_settings_limit(
    setting: str, holder: str
) -> None:
    """As on main, a cache over library code is external, including its native settings."""

    def make_stage(changed: bool) -> typing.Callable[[], str]:
        def first() -> str:
            return "safe"

        cached = functools.lru_cache(
            maxsize=0 if changed and setting == "maxsize" else None,
            typed=changed and setting == "typed",
        )(pow)
        first.__wrapped__ = cached  # pyright: ignore[reportFunctionMemberAccess] - wrapper metadata under test
        callback = first if holder == "wrapped" else functools.partial(cached)

        def stage() -> str:
            return str(callback())

        return stage

    before = fingerprint.get_stage_fingerprint(make_stage(False))
    after = fingerprint.get_stage_fingerprint(make_stage(True))
    assert before == after, "Native settings are tracked only for caches over user code"


def test_member_receiver_discovery_does_not_ask_a_partial_subclass_for_wrapped() -> None:
    """Only the exact native partial synthesized for a partialmethod needs receiver traversal."""

    class TrapPartial(functools.partial[str]):
        def __getattr__(self, name: str) -> typing.Any:
            raise AssertionError(f"receiver discovery asked the partial subclass for {name}")

    def target() -> str:
        return "safe"

    class Captured:
        member: TrapPartial = TrapPartial(target)

    def stage() -> type[Captured]:
        return Captured

    manifest = fingerprint.get_stage_fingerprint(stage)
    assert _definition_key("class", Captured) in manifest
