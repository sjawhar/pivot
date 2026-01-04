"""AST parsing utilities for code analysis.

Provides functions to extract information from Python Abstract Syntax Trees,
primarily for detecting module.attr usage patterns in fingerprinting.

Example:
    >>> def use_numpy():
    ...     import numpy as np
    ...     return np.array([1, 2, 3])
    >>>
    >>> attrs = extract_module_attr_usage(use_numpy)
    >>> attrs
    [('np', 'array')]
"""

import ast
import inspect
from typing import Any


def extract_module_attr_usage(func: Any) -> list[tuple[str, str]]:
    """Extract module.attr patterns (e.g., 'np.array') from function AST."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    attrs: list[tuple[str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                module_name = node.value.id
                attr_name = node.attr
                attrs.append((module_name, attr_name))
            # Chained attributes (os.path.join) handled by recursion through ast.walk()
            elif isinstance(node.value, ast.Attribute):
                pass

    seen = set()
    unique_attrs = []
    for attr in attrs:
        if attr not in seen:
            seen.add(attr)
            unique_attrs.append(attr)

    return unique_attrs


def get_function_ast(func: Any) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Parse function to AST node."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise ValueError(f"Cannot get source for {func}") from e

    tree = ast.parse(source)

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

    for child in ast.iter_child_nodes(node):
        normalize_ast(child)

    return node
