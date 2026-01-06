# Fingerprinting: Research and Design Rationale

This document explains the research process that led to Pivot's AST + getclosurevars fingerprinting approach. It summarizes what alternatives were tested and why they were rejected.

For the complete change detection matrix and implementation details, see [tests/fingerprint/README.md](../tests/fingerprint/README.md).

---

## Approaches Tested

### 1. AST Hashing

**Concept:** Parse function source code into Abstract Syntax Tree, normalize it, and hash the tree structure.

**Pros:**
- Captures semantic structure, ignores formatting
- Can normalize docstrings, function names, whitespace
- Works well for same-module helper functions
- Platform-independent

**Cons:**
- Cannot detect global variable VALUE changes (only structure)
- Cannot detect closure variable VALUE changes
- Cannot detect method call targets (`obj.method()`)
- Cannot detect dynamic calls (`getattr`, `exec`, `eval`)
- Variable name changes affect hash (by design)

**Result:** ✅ **CHOSEN** - Combined with `getclosurevars()` for dependency tracking

---

### 2. Bytecode Hashing

**Concept:** Hash the compiled bytecode (`func.__code__.co_code`) instead of source.

**Pros:**
- Faster than AST parsing
- Works for functions without source code
- Captures compiled behavior

**Cons:**
- Platform/Python-version specific
- **Critical discovery:** `co_code` doesn't include constants!
  ```python
  def func_v1(x): return x + 1
  def func_v2(x): return x + 999

  func_v1.__code__.co_code == func_v2.__code__.co_code  # TRUE!
  ```
  The bytecode only has `LOAD_CONST <index>`; actual values are in `co_consts` tuple.
- Same limitations as AST for closures/globals
- Harder to debug differences

**Result:** ⚠️ **Used as fallback** - With `marshal.dumps(func.__code__)` to capture full code object including constants

---

### 3. cloudpickle

**Concept:** Use cloudpickle to serialize functions and hash the pickle bytes.

**Pros:**
- Seemed simple for collection fingerprinting
- Handles closures automatically
- Works for anonymous functions

**Cons:**
- **Fatal flaw:** Pickles module functions by reference (~34 bytes: module name + function name)
  ```python
  import cloudpickle
  pickled = cloudpickle.dumps(helpers.helper)
  # Result: Just stores "helpers.helper", NOT the bytecode!
  # If helper() changes, pickle hash is IDENTICAL
  ```
- Workaround exists (`cloudpickle.register_pickle_by_value(module)`) but defeats simplicity
- Captures things we DON'T want: docstring/whitespace/name changes
- Doesn't capture transitive dependencies via globals

**Result:** ❌ **REJECTED** - Fundamentally unsuitable for change detection

---

### 4. marshal.dumps()

**Concept:** Use Python's `marshal` module to serialize code objects.

**Pros:**
- Cross-process deterministic
- Detects constant changes (includes `co_consts`)
- Detects logic changes
- Fast and lightweight

**Cons:**
- **Doesn't capture transitive dependencies:**
  ```python
  def leaf(x):
      return x + 1

  def wrapper(x):
      return leaf(x) * 2  # Global reference, NOT closure!
  ```
  If `leaf()` changes, `marshal.dumps(wrapper.__code__)` stays the same because `wrapper` has no closure.

**Result:** ✅ **Used for bytecode fallback only** - Good for individual functions, but misses dependency chain

---

### 5. Comprehensive Hashing (globals + closures + defaults)

**Concept:** Hash everything - AST + global values + closure values + default arguments.

**Pros:**
- Could detect global variable VALUE changes
- Could detect closure variable VALUE changes
- Could detect default argument mutations
- Most accurate representation

**Cons:**
- Performance cost - hashing large objects expensive
- Over-sensitive - triggers on globals updated but not used
- Complex objects may not `repr()` cleanly
- Overkill for most use cases

**Result:** ❌ **Documented as limitation** - Tradeoff not worth it; users can use `pivot clean <stage>` if needed

---

## The Winner: AST + getclosurevars

**Winning combination:**
1. **AST hashing** for stable semantic structure (ignores formatting, normalizes names)
2. **`inspect.getclosurevars()`** to discover dependencies automatically
3. **Recursive fingerprinting** for transitive dependencies

**Why this works:**

`getclosurevars()` captures:
- Top-level imports (both `from X import Y` and `import X` styles)
- Same-module helper functions
- Global constants
- Nonlocal (closure) variables

For module attributes (`import math; math.pi`), we:
1. Detect the module object via `getclosurevars()`
2. AST-scan the function body to find which attributes are accessed
3. Hash those specific attributes (or mark as "callable" for stdlib)

**This automatically handles transitive dependencies:**
- Helper function changes detected → triggers re-fingerprinting
- Transitive helper changes detected recursively
- Works across module boundaries

---

## Implementation Notes

### Bytecode Fallback

When source code unavailable:
```python
if hasattr(func, "__code__"):
    return xxhash.xxh64(marshal.dumps(func.__code__)).hexdigest()
```
Uses `marshal.dumps()` NOT just `co_code` to capture constants.

### Collection Tracking

For dispatch patterns like `FUNCS = {'add': add, 'mul': mul}`:
1. Detect collection in globals
2. Scan for callable values
3. Recursively fingerprint each callable with `get_stage_fingerprint()`
4. Sort dict keys/sets for deterministic ordering

### Class Tracking

Classes tracked with `class:` prefix:
- Direct class imports: `class:MyProcessor`
- Class instances: `class:processor.__class__`
- Full class AST hashed (includes all methods)

### Normalization

AST normalization for stable hashing:
- Function names → `"func"`
- Docstrings → removed
- Whitespace → ignored (AST doesn't capture it)
- Comments → ignored (AST doesn't capture them)

---

## Known Limitations

These are architectural limitations documented in [tests/fingerprint/README.md](../tests/fingerprint/README.md):

1. **Lazy imports inside function body** - `getclosurevars()` only sees module-level bindings
2. **Instance method calls** (`obj.method()`) - Would require type inference
3. **Dynamic patterns** (`getattr(module, name)`, `eval()`) - String values unknown statically
4. **Global variable VALUE changes** - Only structure tracked, not runtime values
5. **Third-party package versions** - Out of scope (use dependency lockfiles)

**Escape hatches:**
- Manual cache invalidation: `pivot clean <stage>`
- Stage-level version parameters
- Module-level imports (workaround for lazy imports)

---

## Comparison Table

| Approach | Deterministic | Detects Constants | Transitive Deps | Simple | Verdict |
|----------|--------------|-------------------|-----------------|--------|---------|
| AST + getclosurevars | ✅ | ✅ | ✅ | ⚠️ | ✅ CHOSEN |
| cloudpickle (default) | ✅ | ❌ | ❌ | ✅ | ❌ REJECTED |
| cloudpickle (by_value) | ✅ | ✅ | ❌ | ❌ | ❌ REJECTED |
| marshal.dumps() | ✅ | ✅ | ❌ | ✅ | ⚠️ Fallback only |
| Bytecode (co_code) | ✅ | ❌ | ❌ | ✅ | ❌ REJECTED |
| Comprehensive hashing | ✅ | ✅ | ✅ | ❌ | ❌ Too complex |

---

## Comparison with Other Tools

No existing pipeline tool tracks transitive Python code dependencies:

| Tool | Code Change Detection |
|------|----------------------|
| **DVC** | Hashes shell command string only |
| **Kedro** | No incremental execution based on code |
| **Hamilton** | No persistent caching between runs |
| **Prefect** | Caches by inputs, not code changes |
| **Dagster** | Hashes op config, not code |
| **Pivot** | AST + getclosurevars with transitive tracking |

---

## Why xxhash64?

Pivot uses xxhash64 throughout for consistency and performance:

- **Fast:** 10-20x faster than SHA-256 for file hashing (hot path)
- **Sufficient:** 64-bit collision resistance adequate for change detection
- **Consistent:** Same algorithm for file content and code fingerprints
- **Non-cryptographic:** No security requirements - just detecting changes

**Output format:** 16-character hexadecimal string (64 bits)

**Collision risk:** Negligible for typical usage (need 4+ billion items for 50% collision probability). Even large ML projects with millions of files and thousands of functions are well within safe margins.

---

## Further Reading

- **Complete change detection matrix:** [tests/fingerprint/README.md](../tests/fingerprint/README.md)
- **Test suite:** [tests/fingerprint/](../tests/fingerprint/)
  - `test_fingerprint.py` - 70 unit tests
  - `test_change_detection.py` - 31 behavior tests
  - `test_integration.py` - 7 end-to-end tests
