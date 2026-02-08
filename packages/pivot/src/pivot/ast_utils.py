import ast
import contextlib
import inspect
import textwrap
import weakref
from collections.abc import Callable
from typing import Any

# Cache for extract_module_attr_usage results using weak references.
# This avoids repeated AST parsing for the same function when processing module dependencies.
# Note: WeakKeyDictionary is not thread-safe. Fingerprinting runs single-threaded
# per process (multiprocessing uses separate memory spaces), so this is safe.
_module_attr_cache: weakref.WeakKeyDictionary[Callable[..., Any], list[tuple[str, str]]] = (
    weakref.WeakKeyDictionary()
)


def extract_module_attr_usage(func: Callable[..., Any]) -> list[tuple[str, str]]:
    """Extract module.attr patterns (e.g., 'np.array') from function AST."""
    # WeakKeyDictionary raises TypeError for non-weakly-referenceable functions (builtins)
    with contextlib.suppress(TypeError):
        cached = _module_attr_cache.get(func)
        if cached is not None:
            return cached

    result = _extract_module_attr_usage_impl(func)

    with contextlib.suppress(TypeError):
        _module_attr_cache[func] = result

    return result


def _extract_module_attr_usage_impl(func: Callable[..., Any]) -> list[tuple[str, str]]:
    """Internal implementation of extract_module_attr_usage."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return []

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return []

    attrs = list[tuple[str, str]]()

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = [node.attr]
            current = node.value

            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                module_name = current.id
                # Reverse chain to get correct order (module.sub.attr)
                attr_path = ".".join(reversed(chain))
                attrs.append((module_name, attr_path))

    return list(dict.fromkeys(attrs))


def get_function_ast(func: Callable[..., Any]) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Parse function to AST node."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise ValueError(f"Cannot get source for {func}") from e

    tree = ast.parse(textwrap.dedent(source))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node

    raise ValueError(f"No FunctionDef found in source for {func}")


def normalize_ast(node: ast.AST) -> ast.AST:
    """Remove docstrings and metadata for stable AST comparison."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body = node.body[1:]
        # Add Pass() if body becomes empty (functions must have non-empty body)
        if not node.body:
            node.body = [ast.Pass()]

    for child in ast.iter_child_nodes(node):
        normalize_ast(child)

    return node
