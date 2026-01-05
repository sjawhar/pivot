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
- Exceptions: type hints in `TYPE_CHECKING` blocks, unambiguous stdlib (`Path`, `dataclass`).

```python
# Good
from pivot import fingerprint
fp = fingerprint.get_stage_fingerprint(func)

# Bad
from pivot.fingerprint import get_stage_fingerprint
fp = get_stage_fingerprint(func)  # Where is this from?
```

## Docstrings

**Simple functions (<20 lines) get one-line docstrings. Period.**

Skip Args/Returns/Examples if type hints make it obvious. Don't repeat what the function signature already says.

```python
# Bad - verbose, repeats type hints
def resolve_path(path: str) -> Path:
    """Resolve path relative to project root (or use absolute).

    Args:
        path: File path (relative or absolute)

    Returns:
        Resolved absolute path
    """

# Good - concise, says what it does
def resolve_path(path: str) -> Path:
    """Resolve relative path from project root; absolute paths unchanged."""

# Bad - explains obvious behavior
def get_project_root() -> Path:
    """Get the project root directory.

    Returns the cached project root if available, otherwise finds it
    by calling find_project_root() and caches the result.

    Returns:
        Path: The project root directory path
    """

# Good - concise
def get_project_root() -> Path:
    """Get project root (cached after first call)."""
```

## Comments - WHY Not WHAT

```python
# Good - explains WHY
# Normalize names to "func" so identical logic produces same hash
node.name = "func"

# Bad - states obvious WHAT
# Parse to AST
tree = ast.parse(source)
```

## Early Returns (Reduce Nesting)

- Use early `return`/`continue` to keep main logic at top indentation level.
- Avoid pyramid of doom; each guard clause should be simple, independent.

## Private Functions

- Use `_prefix` for module-internal helpers; public = exported in `__all__` or used by other modules.

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
