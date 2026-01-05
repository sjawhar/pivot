# Pivot - Project Rules

**Python Version:** 3.13+ | **Coverage Target:** 90%+

---

## Core Design

- Pivot eliminates DVC bottlenecks via per-stage lock files (32x faster), automatic code fingerprinting, and warm worker pools.
- TDD: write tests BEFORE implementation; 90%+ code coverage required.

## Code Quality Standards

- No code duplication; type hints everywhere; `ruff format` (line length 100); `ruff check` for linting.
- Concise one-line docstrings preferred; comments explain WHY not WHAT.
- Private functions use leading underscore; import modules not functions (Google style).
- No circular dependencies; higher-level modules depend on lower-level.

## Linting/Type Config (Critical)

- NEVER modify linting/type rules in `pyproject.toml` without explicit permission.
- No `# type: ignore` without justification; fix code, don't silence checkers.

## Python 3.13+ Type Hints (Critical)

- Constructor syntax for empty collections: `list[int]()` not `: list[int] = []`
- Simplified Generator: `Generator[int]` not `Generator[int, None, None]`
- Use `Callable` instead of `Any` for callables; `Any` only as last resort (dynamic introspection, JSON parsing, testing).
- Use `TypeVar` for type-preserving decorators (preserves exact function signature).
- Use `Callable[..., Any]` for function params, not bare `Any`.
- Document why when using `Any`.

## Import Style (Google Python Style Guide)

- Import modules, not functions: `from pivot import fingerprint` then `fingerprint.get_stage_fingerprint(...)`.
- No relative imports; no `sys.path` modifications.
- No exceptions for stdlib—use `import pathlib` then `pathlib.Path`, not `from pathlib import Path`.
- Exception: type hints in `TYPE_CHECKING` blocks may import types directly.

```python
# Good
from pivot import fingerprint
import pathlib

fp = fingerprint.get_stage_fingerprint(func)
path = pathlib.Path("/some/path")

# Bad
from pivot.fingerprint import get_stage_fingerprint
from pathlib import Path

fp = get_stage_fingerprint(func)  # Where is this from?
path = Path("/some/path")  # Where is Path from?
```

## No `__all__` Declarations

- Don't use `__all__` in modules—it's unnecessary maintenance overhead.
- Use underscore prefix (`_helper`) to indicate private functions.
- For package `__init__.py` re-exports, use explicit re-export syntax: `from module import X as X`.

## Docstrings

**Simple functions (<20 lines) get one-line docstrings. Period.**

Skip Args/Returns/Examples if type hints make it obvious. Don't repeat what the function signature already says.

```python
# Bad - verbose, repeats type hints
def resolve_path(path: str) -> pathlib.Path:
    """Resolve path relative to project root (or use absolute).

    Args:
        path: File path (relative or absolute)

    Returns:
        Resolved absolute path
    """

# Good - concise, says what it does
def resolve_path(path: str) -> pathlib.Path:
    """Resolve relative path from project root; absolute paths unchanged."""

# Bad - explains obvious behavior
def get_project_root() -> pathlib.Path:
    """Get the project root directory.

    Returns the cached project root if available, otherwise finds it
    by calling find_project_root() and caches the result.

    Returns:
        Path: The project root directory path
    """

# Good - concise
def get_project_root() -> pathlib.Path:
    """Get project root (cached after first call)."""
```

## Comments - Only When Necessary

**When to add comments:**

- Non-obvious WHY (e.g., "Validate BEFORE normalizing" not "Validate paths")
- Important timing/ordering (e.g., "Reverse chain to get correct order")
- Known limitations (e.g., "KNOWN ISSUE: lambdas use id() which is non-deterministic")
- TODOs with context (e.g., "TODO: Use threading.Lock for parallel fingerprinting")
- Complex algorithm explanation (e.g., "Case 1: parent contains child")

**When NOT to add comments:**

- Obvious WHAT the code does (e.g., "Add node to graph" before `graph.add_node()`)
- Step-by-step descriptions (e.g., "Step 1:", "Step 2:")
- Redundant with function/variable names (e.g., "Build output map" before `_build_outputs_map()`)
- Information already in docstrings

```python
# Good - explains non-obvious WHY
# Validate paths BEFORE normalizing (check ".." on original paths)
_validate_stage_registration(stages, name, deps, outs)

# Bad - states obvious WHAT
# Add all stages as nodes
for stage in stages:
    graph.add_node(stage)

# Good - clarifies intent
# Reverse chain to get correct order (module.sub.attr)
attr_path = ".".join(reversed(chain))

# Bad - redundant with code
# Parse source code to AST
tree = ast.parse(source)
```

**Rule of thumb:** If removing the comment makes the code unclear, try to write clearer code instead (helper functions, variable names). If that doesn't work, keep the comment.

## Early Returns (Reduce Nesting)

- Use early `return`/`continue` to keep main logic at top indentation level.
- Avoid pyramid of doom; each guard clause should be simple, independent.

## Private Functions

- Use `_prefix` for module-internal helpers; public = used by other modules.

## Error Handling

```python
# Good - specific exception
class FingerprintError(Exception): pass

# Bad - generic
raise Exception("failed")
```

## Logging

```python
# Good
logger.info(f"Stage {name} completed in {duration:.2f}s")

# Bad
print(f"Stage {name} completed")
```

## Development Commands

```bash
uv sync                    # Install dependencies
uv run pytest tests/       # Run tests
uv run ruff format .       # Format
uv run ruff check .        # Lint
uv run basedpyright .      # Type check
```

## Before Returning to User (Critical)

- Must run all four: `ruff format .`, `ruff check .`, `basedpyright .`, `pytest tests/`
- Never say "done" without running these first.

## Critical Discoveries

1. Test helpers must be module-level, not inline—`getclosurevars()` doesn't see module imports in inline closures.
2. Helper functions must NOT start with underscore—fingerprinting filters `_globals` to skip `__name__`, `__file__`, etc., which also filters `_helper()`.
3. Use assertion messages instead of inline comments—messages appear in failure output.
4. **Circular import resolution:** When two modules have bidirectional dependencies, extract shared types/exceptions to a separate module (e.g., `exceptions.py` breaks `registry.py` ↔ `trie.py` cycle).
5. **AST manipulation validation:** After transforming AST nodes (e.g., removing docstrings), validate structural invariants. Function/class bodies must contain at least one statement—add `ast.Pass()` if empty.
6. **Path overlap detection requires Trie:** Simple string matching can't detect directory/file overlaps (`data/` vs `data/train.csv`). Use prefix-based data structure (pygtrie) for comprehensive validation.
