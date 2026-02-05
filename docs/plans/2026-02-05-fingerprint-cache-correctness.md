# Fingerprint Cache Correctness Fixes

**Date:** 2026-02-05  
**Status:** Approved

## Problem Statement

The persistent AST hash cache in `src/pivot/fingerprint.py` has several correctness issues:

1. **Lambda cache key collision** — Multiple lambdas in the same file all have `__qualname__ == "<lambda>"`, causing them to collide in the cache and potentially return wrong hashes.

2. **Python version not in cache key** — Upgrading Python can change AST representation and bytecode format, but cached hashes persist, causing stale cache hits.

3. **Indentation breaks AST parsing** — `inspect.getsource()` returns indented source for methods/nested functions. `ast.parse()` fails with `IndentationError`, falling back to raw source hashing that is sensitive to whitespace/comments.

## Solution Overview

1. Disambiguate lambda qualnames with line and column numbers
2. Add Python version and schema version to cache keys
3. Dedent source before AST parsing
4. Add `pivot fingerprint reset` CLI command

## Design

### Cache Key Changes

Current cache key:
```
(rel_path, mtime_ns, size, inode, qualname)
```

New cache key:
```
(rel_path, mtime_ns, size, inode, qualname, py_version, schema_version)
```

**New fields:**

- **`py_version`**: `f"{sys.version_info.major}.{sys.version_info.minor}"` — invalidates on Python upgrade
- **`schema_version`**: Integer constant starting at `1` — allows cache invalidation when fingerprinting algorithm changes

### Lambda Disambiguation

Transform qualname for lambdas to include position:

```python
def _get_qualname_for_cache(func: Callable[..., Any]) -> str:
    """Get qualname, disambiguated for lambdas."""
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "<unknown>")
    
    if "<lambda>" not in qualname:
        return qualname
    
    code = getattr(func, "__code__", None)
    if code is None:
        return qualname
    
    lineno = code.co_firstlineno
    col = 0
    if hasattr(code, "co_positions"):
        for _, _, c, _ in code.co_positions():
            if c is not None:
                col = c
                break
    
    return f"{qualname}:{lineno}:{col}"
```

Result:
- Normal function: `"my_function"` (unchanged)
- Lambda: `"<lambda>:42:8"` or `"make_stage.<locals>.<lambda>:15:12"`

Named functions don't include line numbers, so moving them in a file doesn't cause cache misses.

### Indentation Fix

In `_compute_function_hash()`, dedent source before parsing:

```python
import textwrap

try:
    source = inspect.getsource(func)
except (OSError, TypeError):
    # ... existing fallback to __code__ or id() ...

try:
    tree = ast.parse(textwrap.dedent(source))
except SyntaxError:
    # Fallback: hash dedented source (not raw source)
    return xxhash.xxh64(textwrap.dedent(source).encode()).hexdigest()
```

On parse failure, we hash *dedented* source — this at least normalizes leading indentation even when AST parsing fails for edge cases.

### Constants

```python
import sys

_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
_CACHE_SCHEMA_VERSION = 1
```

Bump `_CACHE_SCHEMA_VERSION` whenever the fingerprinting algorithm changes in ways that affect hash output.

### CLI Command

```bash
pivot fingerprint reset  # Clears cached function hashes from StateDB
```

Implementation:
- Delete all entries with `ast:` prefix from StateDB
- Print count of cleared entries

Help text:
```
Reset cached function fingerprints. Use after encountering stale cache issues
or when troubleshooting unexpected stage re-runs.
```

## File Changes

### Modified

1. **`src/pivot/fingerprint.py`**
   - Add `_PYTHON_VERSION` and `_CACHE_SCHEMA_VERSION` constants
   - Add `_get_qualname_for_cache()` helper for lambda disambiguation
   - Update `hash_function_ast()` to use new cache key fields
   - Update `_compute_function_hash()` to dedent source before parsing

2. **`src/pivot/storage/state.py`**
   - Update `get_ast_hash()` signature and key construction
   - Update `save_ast_hash_many()` signature and key construction
   - Add `clear_ast_hashes()` method for the reset command

3. **`src/pivot/cli/__init__.py`** (or submodule)
   - Add `pivot fingerprint reset` command

### New Tests

4. **`tests/fingerprint/test_fingerprint.py`**
   - Test lambda disambiguation (two lambdas same file, different hashes)
   - Test same-line lambdas disambiguated by column
   - Test method/nested function comment change → same hash

5. **`tests/fingerprint/test_determinism.py`**
   - Test Python version in cache key
   - Test schema version invalidation

6. **`tests/cli/test_fingerprint_cli.py`**
   - Test `pivot fingerprint reset` clears StateDB entries

## Testing Strategy

### Lambda Disambiguation

```python
def test_multiple_lambdas_same_file_different_hashes(module_dir):
    """Two lambdas on different lines get different cache keys."""
    mod_py = module_dir / "test_lambdas.py"
    mod_py.write_text("""
lambda_a = lambda x: x + 1
lambda_b = lambda x: x + 2
""")
    mod = _import_fresh("test_lambdas")
    
    h1 = fingerprint.hash_function_ast(mod.lambda_a)
    h2 = fingerprint.hash_function_ast(mod.lambda_b)
    
    assert h1 != h2
    q1 = fingerprint._get_qualname_for_cache(mod.lambda_a)
    q2 = fingerprint._get_qualname_for_cache(mod.lambda_b)
    assert q1 != q2


def test_same_line_lambdas_disambiguated(module_dir):
    """Two lambdas on same line get different cache keys via column."""
    mod_py = module_dir / "test_same_line.py"
    mod_py.write_text("pair = (lambda x: x, lambda y: y)\n")
    mod = _import_fresh("test_same_line")
    
    q1 = fingerprint._get_qualname_for_cache(mod.pair[0])
    q2 = fingerprint._get_qualname_for_cache(mod.pair[1])
    assert q1 != q2
```

### Dedent Fix

```python
def test_method_comment_change_no_miss(module_dir):
    """Changing comments in methods should not cause cache miss."""
    mod_py = module_dir / "test_method.py"
    mod_py.write_text("""
class MyClass:
    def method(self):
        # original comment
        return 42
""")
    mod = _import_fresh("test_method")
    h1 = fingerprint.hash_function_ast(mod.MyClass.method)
    
    mod_py.write_text("""
class MyClass:
    def method(self):
        # CHANGED comment
        return 42
""")
    mod = _import_fresh("test_method")
    h2 = fingerprint.hash_function_ast(mod.MyClass.method)
    
    assert h1 == h2
```

## Migration

No explicit migration needed:

1. Schema version `1` is now in cache keys
2. Old entries (no schema version) won't match → cache misses
3. New entries accumulate with correct keys
4. Old entries are harmless (just take space)

Users can run `pivot fingerprint reset` to clean up if desired.

## Documentation Updates

1. **`tests/fingerprint/README.md`** — Update change detection matrix:
   - Add row for Python version upgrade → cache invalidated
   - Add note about lambda disambiguation

2. **`docs/solutions/`** — Add fingerprint cache troubleshooting doc

## Changelog

```
### Fixed
- Lambda functions in same file no longer collide in fingerprint cache
- Methods and nested functions now correctly ignore comment/whitespace changes
- Python version upgrades now invalidate fingerprint cache
```
