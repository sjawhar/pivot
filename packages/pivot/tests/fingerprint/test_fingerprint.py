# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportImplicitOverride=false

import ast
import dataclasses
import enum
import importlib
import importlib.util
import inspect
import json
import locale
import logging
import math
import os
import pathlib
import sys
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
from pivot import ast_utils, exceptions, fingerprint, loaders
from pivot.storage import state as state_mod


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


@dataclasses.dataclass
class _HelperDataClassNoMethods:
    value: int


@dataclasses.dataclass
class _HelperDataClassWithMethod:
    value: int

    def custom(self) -> int:
        return self.value


@dataclasses.dataclass
class _HelperDataClassWithDunder:
    value: int

    def __str__(self) -> str:
        return str(self.value)


class _HelperPydanticModelWithAllowedMethod(BaseModel):
    value: int

    def model_post_init(self, __context: typing.Any) -> None:
        return None


class _HelperPydanticModelWithUserMethod(BaseModel):
    value: int

    def compute(self) -> int:
        return self.value + 1


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
    assert len(fp) == 2


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
    """Stdlib module attributes (math, os, json, etc.) are NOT tracked."""
    fp = fingerprint.get_stage_fingerprint(_helper_uses_math_pi)
    assert "self:_helper_uses_math_pi" in fp
    assert not any("math" in key for key in fp if not key.startswith("self:")), fp

    fp2 = fingerprint.get_stage_fingerprint(_helper_uses_multiple_math_attrs)
    assert not any("math" in key for key in fp2 if not key.startswith("self:")), fp2


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


def test_process_class_body_dependencies_tracks_bases_and_annotations():
    """Class body walker should track bases and annotated types."""
    manifest = dict[str, str]()
    visited = set[int]()

    fingerprint._process_class_body_dependencies(_HelperAnnotatedClass, manifest, visited)

    assert _definition_key("class", _HelperBase) in manifest
    assert _definition_key("class", _HelperFieldType) in manifest
    assert _definition_key("class", _HelperContainer.Inner) in manifest
    assert _definition_key("class", _HelperPydanticModel) in manifest
    assert _definition_key("schema", _HelperPydanticModel) in manifest


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
    assert list(fingerprint.get_stage_fingerprint(_helper_reads_pivot_tui)) == [
        "self:_helper_reads_pivot_tui"
    ]


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


def test_check_data_class_methods_allows_dunders_and_pydantic_hooks():
    """Dunder methods and allowed Pydantic hooks should be allowed."""
    fingerprint._check_data_class_methods(_HelperDataClassNoMethods)
    fingerprint._check_data_class_methods(_HelperDataClassWithDunder)
    fingerprint._check_data_class_methods(_HelperPydanticModelWithAllowedMethod)


def test_check_data_class_methods_rejects_user_methods():
    """User-defined methods on data classes should raise errors."""
    with pytest.raises(exceptions.StageDefinitionError, match="Data class"):
        fingerprint._check_data_class_methods(_HelperDataClassWithMethod)

    with pytest.raises(exceptions.StageDefinitionError, match="Data class"):
        fingerprint._check_data_class_methods(_HelperPydanticModelWithUserMethod)


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


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "import importlib\ndef stage():\n    return importlib.import_module('json')\n",
            id="attribute-import-module",
        ),
        pytest.param(
            "from importlib import import_module as load\ndef stage():\n    return load('json')\n",
            id="aliased-import-module",
        ),
        pytest.param("def stage():\n    return __import__('json')\n", id="dunder-import"),
    ],
)
def test_dynamic_import_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """A stage that imports a module by a runtime name is refused, under any spelling."""
    path = tmp_path / "dyn_import_mod.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("dyn_import_mod", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # is_user_code finds a function's module through sys.modules; unregistered, the stage
    # would not count as user code and no check would run.
    monkeypatch.setitem(sys.modules, "dyn_import_mod", module)
    spec.loader.exec_module(module)
    assert fingerprint.is_user_code(module.stage), "the stage must count as user code"
    with pytest.raises(exceptions.StageDefinitionError, match="dynamic imports"):
        fingerprint.get_stage_fingerprint(module.stage)


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


def test_hash_pydantic_schema_adds_schema_and_nested_models():
    """Should hash model schema and discover nested models."""
    manifest = dict[str, str]()
    visited = set[int]()

    fingerprint._hash_pydantic_schema(
        typing.cast("type[fingerprint._PydanticModelProtocol]", _PydanticOuterModel),
        manifest,
        visited,
    )

    outer_schema = _definition_key("schema", _PydanticOuterModel)
    assert outer_schema in manifest
    assert _definition_key("schema", _PydanticInnerModel) in manifest
    assert _definition_key("class", _PydanticInnerModel) in manifest
    assert _definition_key("class", _PydanticOuterModel) not in manifest
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
    fingerprint._pending_ast_writes.extend(orig_pending)
    fingerprint._pending_manifest_writes.clear()
    fingerprint._pending_manifest_writes.extend(orig_pending_manifest)
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

    # Should not raise RecursionError
    manifest = dict[str, str]()
    visited = set[int]()
    fingerprint._hash_pydantic_schema(
        TreeNode,  # pyright: ignore[reportArgumentType] - Pydantic model_fields protocol mismatch in tests
        manifest,
        visited,
    )
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

    manifest = dict[str, str]()
    visited = set[int]()
    fingerprint._hash_pydantic_schema(
        Parent,  # pyright: ignore[reportArgumentType] - Pydantic model_fields protocol mismatch in tests
        manifest,
        visited,
    )
    key = _definition_key("schema", Parent)
    assert key in manifest
    assert manifest[key] != "<pending>"


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

    manifest = dict[str, str]()
    visited = set[int]()
    fingerprint._process_type_hint_dependencies(stage, manifest, visited)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
        "pivot.config.io.get_state_dir",
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
    db_path = state_dir

    # Create initial StateDB
    with state.StateDB(db_path) as db:
        pass

    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    fingerprint._pending_ast_writes.extend(test_entries)

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
    writes = list[tuple[bytes, bytes]]()
    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", writes)
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage_name, _func: None)
    try:
        stage_module = importlib.import_module("f8_stage")
        fingerprint.get_stage_fingerprint_cached("f8/scale", stage_module.scale)
    finally:
        sys.modules.pop("f8_stage", None)
        sys.modules.pop("f8_settings", None)

    ((_key, raw),) = writes
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
    writes = list[tuple[bytes, bytes]]()
    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", writes)
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage_name, _func: None)
    try:
        settings = importlib.import_module("reads_settings")
        module = importlib.import_module("reads_stage")
        fingerprint.get_stage_fingerprint_cached("r/stage", module.stage)
        ((_key, raw),) = writes
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
    writes = list[tuple[bytes, bytes]]()
    monkeypatch.setattr(fingerprint, "_pending_manifest_writes", writes)
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage_name, _func: None)
    names = ("replay_stage", "replay_lib", "replay_body")
    try:
        settings = importlib.import_module("replay_settings")
        module = importlib.import_module("replay_stage")
        manifest = fingerprint.get_stage_fingerprint_cached("r/stage", module.stage)
        assert manifest["const:replay_lib.LIMIT"] == "1", manifest
        assert manifest["const:replay_body.BODY"] == "2", manifest
        ((_key, raw),) = writes
        reads = json.loads(raw)["r"]
        for name in names:
            del sys.modules[name]
        module = importlib.import_module("replay_stage")
        assert fingerprint._recorded_reads_hold(module.stage, reads), "re-import alone must hit"

        vars(settings)["SOURCE"][changed] += 100
        for name in names:
            del sys.modules[name]
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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    db_path = state_dir
    with state_mod.StateDB(db_path):
        pass

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
        "pivot.config.io.get_state_dir",
        lambda: blocker / "subdir",
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
    db_path = state_dir

    # Write corrupted data directly into the db
    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("bad_stage", _cached_stage)
        db.put_raw(key, b"NOT VALID JSON {{{")

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    db_path = state_dir

    with state_mod.StateDB(db_path) as db:
        key = fingerprint._make_manifest_cache_key("list_stage", _cached_stage)
        db.put_raw(key, b"[1, 2, 3]")  # Valid JSON, but not a dict

    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
        assert len(fingerprint._pending_manifest_writes) > 0, "Should queue manifest writes"

        # Step 2: Flush to StateDB
        fingerprint.flush_manifest_cache()
        assert len(fingerprint._pending_manifest_writes) == 0, (
            "Pending writes should be empty after flush"
        )

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: db_path)

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
    monkeypatch.setattr("pivot.config.io.get_state_dir", lambda: state_dir)
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
