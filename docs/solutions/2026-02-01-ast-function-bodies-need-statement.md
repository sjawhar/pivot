---
tags: [python, ast, code-generation]
category: gotcha
module: fingerprint
symptoms: ["SyntaxError when generating code", "invalid AST", "empty function body error"]
---

# AST Manipulation: Function Bodies Require at Least One Statement

## Problem

When manipulating Python AST for fingerprinting (stripping docstrings, removing decorators, normalizing functions), the resulting function body can become empty. Python's grammar requires function bodies to contain at least one statement:

```python
def normalize_ast(node: ast.AST) -> ast.AST:
    """Remove docstrings for stable AST comparison."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # Check if first statement is a docstring
        if (node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]  # Remove docstring
            # BUG: If function was ONLY a docstring, body is now empty!
```

A docstring-only function:

```python
def placeholder():
    """This function intentionally does nothing."""
```

After stripping the docstring, `node.body` becomes an empty list. When you try to compile or unparse this AST, Python raises an error because function bodies cannot be empty.

## Solution

After any operation that removes statements from a function/class body, check if the body is empty and add `ast.Pass()` as a placeholder:

```python
def normalize_ast(node: ast.AST) -> ast.AST:
    """Remove docstrings for stable AST comparison."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if (node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
            # Ensure body is never empty (functions must have non-empty body)
            if not node.body:
                node.body = [ast.Pass()]

    for child in ast.iter_child_nodes(node):
        normalize_ast(child)

    return node
```

This applies to any AST manipulation that removes statements:
- Stripping docstrings
- Removing decorator logic
- Filtering out certain statement types
- Any transformation that could leave a body empty

## Key Insight

Python's grammar requires all compound statements (`def`, `class`, `if`, `for`, `while`, `try`, `with`) to have non-empty bodies. When manipulating AST, always guard against creating empty bodies by inserting `ast.Pass()` as a syntactically valid no-op placeholder.
