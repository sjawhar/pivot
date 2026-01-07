"""Automatic code change detection using Python introspection.

This module implements the core fingerprinting algorithm that detects when user-defined
functions change, enabling automatic pipeline re-execution without manual declarations.

Algorithm:
    1. Hash the function itself using AST
    2. Use inspect.getclosurevars() to find all referenced names
    3. For each reference:
       - If callable and user code → hash and recurse (transitive deps)
       - If module → AST scan for module.attr usage (Google-style imports)
       - If simple constant → capture value
    4. Return manifest dictionary with all dependencies

Validated: See docs/fingerprinting.md and tests/fingerprint/

Example:
    >>> def helper(x):
    ...     return x * 2
    >>>
    >>> def stage_func(data):
    ...     return helper(data) + 1
    >>>
    >>> fp = get_stage_fingerprint(stage_func)
    >>> print(fp)
    {
        'self:stage_func': 'a1b2c3d4...',
        'func:helper': '5e6f7g8h...'
    }
"""

import ast
import inspect
import marshal
import pathlib
import sys
import types
from collections.abc import Callable
from typing import Any, cast

import xxhash

from pivot import ast_utils

_SITE_PACKAGE_PATHS = ("site-packages", "dist-packages")


def get_stage_fingerprint(
    func: Callable[..., Any], visited: set[int] | None = None
) -> dict[str, str]:
    """Generate fingerprint manifest capturing all code dependencies.

    Returns dict with keys:
    - 'self:<name>': Function itself (hash)
    - 'func:<name>': Referenced helper functions (hash, transitive)
    - 'class:<name>': Referenced class definitions (hash, transitive)
    - 'mod:<module>.<attr>': Module attributes (hash for user code, "callable" for stdlib)
    - 'const:<name>': Global constants (repr value)
    """
    if visited is None:
        visited = set()
        # TODO (future): If parallel fingerprinting is needed, use threading.Lock
        # to protect visited set from race conditions. Current single-threaded
        # usage is safe.

    manifest = dict[str, str]()

    func_id = id(func)
    if func_id in visited:
        return manifest
    visited.add(func_id)

    func_name = getattr(func, "__name__", "<lambda>")
    manifest[f"self:{func_name}"] = hash_function_ast(func)

    try:
        closure_vars = inspect.getclosurevars(func)
    except (TypeError, AttributeError):
        return manifest

    for name, value in closure_vars.globals.items():
        if name.startswith("__"):
            continue

        if callable(value) and is_user_code(value):
            _process_callable_dependency(name, value, manifest, visited)
        elif isinstance(value, types.ModuleType):
            _process_module_dependency(name, value, func, manifest, visited)
        elif isinstance(value, (bool, int, float, str, bytes, type(None))):
            manifest[f"const:{name}"] = repr(value)
        elif isinstance(value, (dict, list, tuple, set, frozenset)):
            _process_collection_dependency(
                name,
                cast(
                    "dict[Any, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any]",
                    value,
                ),
                manifest,
                visited,
            )
        elif _is_user_class_instance(value):
            _process_instance_dependency(name, value, manifest, visited)

    for name, value in closure_vars.nonlocals.items():
        if callable(value) and is_user_code(value):
            _process_callable_dependency(name, value, manifest, visited)
        elif isinstance(value, (bool, int, float, str, bytes, type(None))):
            manifest[f"const:{name}"] = repr(value)
        elif isinstance(value, (dict, list, tuple, set, frozenset)):
            _process_collection_dependency(
                name,
                cast(
                    "dict[Any, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any]",
                    value,
                ),
                manifest,
                visited,
            )
        elif _is_user_class_instance(value):
            _process_instance_dependency(name, value, manifest, visited)

    return manifest


def _process_callable_dependency(
    name: str, func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Process a callable dependency and add to manifest."""
    # Use 'class:' prefix for type objects, 'func:' for functions
    prefix = "class" if isinstance(func, type) else "func"
    _add_callable_to_manifest(f"{prefix}:{name}", func, manifest, visited)


def _is_user_class_instance(value: Any) -> bool:
    """Check if value is an instance of a user-defined class."""
    cls = cast("type[Any]", type(value))
    # Skip built-in types and common stdlib types
    if cls.__module__ == "builtins":
        return False
    return is_user_code(cls)


def _process_instance_dependency(
    name: str, instance: Any, manifest: dict[str, str], visited: set[int]
) -> None:
    """Track the class definition of a user-defined instance."""
    cls = cast("type[Any]", type(instance))
    _add_callable_to_manifest(f"class:{name}.__class__", cls, manifest, visited)


def _process_collection_dependency(
    name: str,
    collection: dict[Any, Any] | list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Scan collection for callable user code and add to manifest."""
    if isinstance(collection, dict):
        # Use sorted keys for deterministic ordering
        for key in sorted(collection.keys(), key=_sort_key):
            value = collection[key]
            if callable(value) and is_user_code(value):
                _add_callable_to_manifest(f"func:{name}[{key!r}]", value, manifest, visited)
    else:
        # For sequences and sets, use enumerate for index-based keys
        # Sort sets for deterministic ordering
        items = (
            sorted(collection, key=_sort_key)
            if isinstance(collection, (set, frozenset))
            else collection
        )
        for i, value in enumerate(items):
            if callable(value) and is_user_code(value):
                _add_callable_to_manifest(f"func:{name}[{i}]", value, manifest, visited)


def _sort_key(value: Any) -> tuple[str, str]:
    """Sort key that handles mixed types safely."""
    return (type(value).__name__, str(value))


def _add_callable_to_manifest(
    key: str, func: Callable[..., Any], manifest: dict[str, str], visited: set[int]
) -> None:
    """Hash callable and merge its transitive dependencies into manifest."""
    manifest[key] = hash_function_ast(func)
    for child_key, child_val in get_stage_fingerprint(func, visited).items():
        if not child_key.startswith("self:"):
            manifest[child_key] = child_val


def _process_module_dependency(
    name: str,
    module: types.ModuleType,
    func: Callable[..., Any],
    manifest: dict[str, str],
    visited: set[int],
) -> None:
    """Process module attribute dependencies and add to manifest."""
    attrs = ast_utils.extract_module_attr_usage(func)
    module_name = getattr(module, "__name__", name)

    for mod_name, attr_name in attrs:
        if mod_name not in (name, module_name):
            continue
        key = f"mod:{mod_name}.{attr_name}"
        if key in manifest:
            continue
        try:
            attr_value = getattr(module, attr_name)
        except AttributeError:
            manifest[key] = "unknown"
        else:
            if callable(attr_value) and is_user_code(attr_value):
                _add_callable_to_manifest(key, attr_value, manifest, visited)
            elif callable(attr_value):
                manifest[key] = "callable"
            else:
                manifest[key] = repr(attr_value)


def hash_function_ast(func: Callable[..., Any]) -> str:
    """Hash function AST (ignores whitespace, comments, docstrings).

    Limitation: Lambdas and functions without source code fall back to id(func),
    which is non-deterministic across runs. This causes unnecessary re-runs for
    stages using lambdas. Mitigation: Use named functions instead of lambdas in
    pipeline stages for stable fingerprinting.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        if hasattr(func, "__code__"):
            # marshal.dumps captures full code object including co_consts
            # (co_code alone doesn't include constants - x+1 and x+999 have same co_code!)
            return xxhash.xxh64(marshal.dumps(func.__code__)).hexdigest()
        # KNOWN ISSUE: Using id(func) is non-deterministic across runs
        # This affects lambdas without source code, causing unnecessary re-runs
        return xxhash.xxh64(str(id(func)).encode()).hexdigest()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return xxhash.xxh64(source.encode()).hexdigest()

    tree = _normalize_ast(tree)
    ast_str = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return xxhash.xxh64(ast_str.encode()).hexdigest()


def _normalize_ast(node: ast.AST) -> ast.AST:
    """Remove docstrings and normalize function names for stable hashing."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.name = "func"

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and _has_docstring(
        node
    ):
        node.body = node.body[1:]
        # Ensure body is never empty after removing docstring
        if not node.body:
            node.body = [ast.Pass()]

    for child in ast.iter_child_nodes(node):
        _normalize_ast(child)

    return node


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Check if node has a docstring as first statement."""
    return (
        bool(node.body)
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )


def is_user_code(obj: Any) -> bool:
    """Check if object is user code (not stdlib/site-packages/builtins)."""
    if obj is None:
        return False

    module = _get_module(obj)
    if module is None:
        return False

    # Built-in modules (sys, builtins, _io, etc.) are not user code
    module_name = getattr(module, "__name__", "")
    if module_name in sys.builtin_module_names:
        return False

    # If no __file__, assume user code (exec/notebook/interactive)
    # since stdlib and site-packages always have __file__
    if not hasattr(module, "__file__") or module.__file__ is None:
        return True

    module_file = pathlib.Path(module.__file__).resolve()

    if _is_stdlib_path(module_file):
        return False

    return not any(path in module_file.parts for path in _SITE_PACKAGE_PATHS)


def _get_module(obj: Any) -> types.ModuleType | None:
    """Get module for an object, handling both modules and module members."""
    if isinstance(obj, types.ModuleType):
        return obj

    if not hasattr(obj, "__module__"):
        return None

    module_name = obj.__module__
    if module_name == "builtins":
        return None

    return sys.modules.get(module_name)


def _is_stdlib_path(module_file: pathlib.Path) -> bool:
    """Check if path is in Python stdlib (but not site-packages)."""
    stdlib_paths = [pathlib.Path(sys.prefix), pathlib.Path(sys.base_prefix)]

    for stdlib_path in stdlib_paths:
        try:
            if stdlib_path in module_file.parents:
                return not any(path in module_file.parts for path in _SITE_PACKAGE_PATHS)
        except (ValueError, OSError):
            continue

    return False
