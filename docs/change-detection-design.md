# Pivot Change Detection: Design Document

**Status**: RFC (Request for Comments)  
**Authors**: [Your name]  
**Reviewers**: [Colleagues to review]  
**Last Updated**: 2026-01-04

---

## Executive Summary

This document outlines our approach to detecting when a pipeline stage needs to re-run. The core challenge is determining when a Python function's **behavior** has changed, which is harder than it sounds because Python is dynamic and functions can depend on many things beyond their immediate source code.

We propose using **AST-based function hashing** combined with **lazy import tracking** as our primary mechanism, with explicit fallbacks for edge cases.

**We are seeking feedback on this approach before implementation.**

---

## Problem Statement

In a pipeline tool, we want to skip stages that don't need to run. A stage should re-run when:

1. Its input files changed
2. Its parameters changed
3. Its **code** changed
4. An upstream stage was re-run
5. Its outputs are missing

Items 1, 2, 4, and 5 are straightforward to detect (file hashes, param hashes, run IDs, existence checks).

**Item 3 (code changes) is the hard problem this document addresses.**

### Why Is Code Change Detection Hard?

A Python function's behavior can depend on:

```python
GLOBAL_THRESHOLD = 0.5  # Global variable

def helper(x):  # Helper function in same module
    return x * 2

@stage(deps=["data.csv"], outs=["out.csv"])
def train(deps, outs, params):
    from external_module import process  # Cross-module dependency
    import pandas as pd  # Third-party library

    df = pd.read_csv(deps[0])
    df = helper(df)  # Uses helper
    df = process(df)  # Uses external

    if df.max() > GLOBAL_THRESHOLD:  # Uses global
        ...
```

Changes to ANY of these could affect behavior:

- The function body itself
- `helper()` function
- `process()` in another module
- `GLOBAL_THRESHOLD` variable
- The `pandas` library version

**We cannot perfectly track all of these.** The question is: what can we reasonably track, and how do we document the limitations?

---

## Options Considered

### Option 1: Hash Only the Function Body (Baseline)

**How it works:**

```python
import ast, inspect

def get_function_hash(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    return hash(ast.dump(tree))
```

**What it catches:**

- Changes to the function's own code
- Changes to arguments and defaults

**What it misses:**

- Helper functions (same module)
- Cross-module dependencies
- Global variables
- Third-party package changes

**Verdict**: Too limited. Users will be confused when helper function changes don't trigger re-runs.

---

### Option 2: Hash the Entire Module

**How it works:**

```python
def get_module_hash(func):
    module = sys.modules[func.__module__]
    source = inspect.getsource(module)
    return hash(source)
```

**What it catches:**

- Everything in the same module (helpers, globals, etc.)

**What it misses:**

- Cross-module dependencies
- Third-party packages

**Problems:**

- Over-triggers: changing an unrelated function in the module triggers all stages
- Large modules = slow hashing

**Verdict**: Too coarse. Creates unnecessary re-runs.

---

### Option 3: Transitive Hashing (Same Module Only)

**How it works:**

1. Parse the function's AST
2. Find all function calls
3. Resolve calls to same-module functions
4. Recursively hash those functions
5. Combine all hashes

```python
def get_transitive_hash(func, visited=None):
    if visited is None:
        visited = set()
    if id(func) in visited:
        return ""
    visited.add(id(func))

    hashes = [get_function_hash(func)]

    for called_name in extract_called_functions(func):
        called_func = resolve_in_same_module(func, called_name)
        if called_func:
            hashes.append(get_transitive_hash(called_func, visited))

    return combined_hash(hashes)
```

**What it catches:**

- Function body
- Helper functions in same module (transitively)

**What it misses:**

- Cross-module dependencies
- Global variables (could add)
- Third-party packages

**Verdict**: Good baseline. Catches the common case of helper functions.

---

### Option 4: Lazy Import Convention + AST Tracking

**How it works:**

Encourage users to use lazy imports inside function bodies:

```python
@stage(...)
def train(deps, outs, params):
    from myutils import clean_data  # <-- In function body, visible to AST
    from myutils import validate

    df = clean_data(deps[0])
    validate(df)
```

Then parse the AST to find `from X import Y` statements and hash those functions:

```python
def get_lazy_import_hash(func):
    imports = extract_import_statements(func)  # AST parsing
    hashes = []

    for module_name, func_name in imports:
        if is_user_module(module_name):  # Not stdlib/third-party
            module = importlib.import_module(module_name)
            imported_func = getattr(module, func_name)
            hashes.append(get_function_hash(imported_func))

    return combined_hash(hashes)
```

**What it catches:**

- Function body
- Same-module helpers (via transitive hashing)
- Cross-module functions IF they're lazy-imported

**What it requires:**

- Users adopt lazy import style for user utilities
- Third-party imports can stay at top level (they're preloaded anyway)

**Verdict**: Elegant solution that makes dependencies explicit without a separate declaration.

---

### Option 5: Explicit `code_deps` Parameter

**How it works:**

Users explicitly declare code dependencies:

```python
from myutils import clean_data, validate

@stage(
    deps=["data.csv"],
    outs=["out.csv"],
    code_deps=[clean_data, validate],  # Explicit list
)
def train(deps, outs, params):
    df = clean_data(deps[0])
    validate(df)
```

**What it catches:**

- Everything the user declares

**Problems:**

- Easy to forget to update when adding new calls
- Duplication (function used in body AND in code_deps)
- Can get out of sync

**Verdict**: Works but error-prone. Better as fallback than primary mechanism.

---

### Option 6: Python Files as Dependencies

**How it works:**

Treat Python source files as regular file dependencies:

```python
@stage(
    deps=["data.csv", "src/utils.py", "src/preprocessing.py"],
    outs=["out.csv"],
)
def train(deps, outs, params):
    from src.utils import clean_data
    ...
```

**What it catches:**

- Any change to the listed Python files

**Problems:**

- Verbose
- Easy to forget
- Over-triggers if file has unrelated changes

**Verdict**: Good explicit fallback for complex cases.

---

### Option 7: Git-Based Change Detection

**How it works:**

Track which commit the stage was last run at, use git to detect file changes:

```python
def has_code_changed(func, last_run_commit):
    module_file = inspect.getfile(func)
    return git_file_changed_since(module_file, last_run_commit)
```

**What it catches:**

- Any change to the module file

**Problems:**

- Requires git
- What about uncommitted changes?
- Over-triggers (any change to file, not just relevant functions)

**Verdict**: Interesting but too coarse and adds git dependency.

---

### Option 8: Static Call Graph Analysis

**How it works:**

Use a tool like `pycg` to build a complete call graph:

```python
def get_all_reachable_functions(func):
    call_graph = pycg.analyze(func)
    return call_graph.get_all_callees(func)
```

**What it catches:**

- Theoretically everything reachable

**Problems:**

- Can't handle dynamic calls (`getattr`, `func_dict[key]()`)
- Complex to implement correctly
- May be slow for large codebases
- External dependency

**Verdict**: Overkill for our use case. Might revisit later.

---

## Our Recommendation

### Primary Mechanism: Option 3 + Option 4

Combine **transitive same-module hashing** with **lazy import tracking**:

1. Hash the stage function itself
2. Transitively hash all same-module functions it calls
3. Parse function body for `from X import Y` statements
4. Hash those imported user-defined functions

### Fallback: Option 6

For complex cases, allow explicit Python file dependencies:

```python
@stage(deps=["data.csv", "src/legacy_utils.py"], outs=["out.csv"])
def train(deps, outs, params):
    ...
```

### Recommended Coding Convention

```python
@stage(deps=["data.csv"], outs=["out.csv"])
def train(deps, outs, params: TrainParams):
    # Third-party: can be at top of file (preloaded in warm workers)
    import pandas as pd
    import numpy as np

    # User utilities: lazy import so AST can track them
    from src.utils import clean_data
    from src.preprocessing import normalize

    df = pd.read_csv(deps[0])
    df = clean_data(df)
    df = normalize(df)
    ...
```

### Escape Hatches

```bash
pivot run --force train      # Force specific stage
pivot run --force-all        # Force everything
pivot run --explain          # Show why each stage will/won't run
```

---

## Change Detection Matrix

### What WILL Trigger Re-run

| Change Type                       | Mechanism            |
| --------------------------------- | -------------------- |
| Stage function body               | AST hash             |
| Stage function arguments/defaults | AST hash             |
| Same-module helper function       | Transitive hash      |
| Lazy-imported user function       | Lazy import tracking |
| Parameter values                  | Params hash          |
| Env vars (via params defaults)    | Params hash          |
| Input file content                | File hash            |
| Upstream stage re-ran             | Run ID check         |
| Output file missing               | Existence check      |
| Explicit `deps=["file.py"]`       | File hash            |

### What WILL NOT Trigger Re-run

| Change Type                      | Workaround                       |
| -------------------------------- | -------------------------------- |
| Top-level imported user function | Use lazy import                  |
| Third-party package version      | Optional `package_deps` (future) |
| Global variable value            | Move to params                   |
| Closure variable                 | Move to params                   |
| Class instance attribute         | Move to params                   |
| `exec()`/`eval()` dynamic code   | Don't use these                  |

---

## Implementation Details

### AST Normalization

We normalize the AST before hashing to avoid false positives:

**Remove:**

- Docstrings (don't affect behavior)

**Keep:**

- Variable names (renaming is rare, users expect re-run)
- String/number literals (definitely affect behavior)
- Comments are already stripped by parser

```python
class DocstringRemover(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if self._is_docstring_position(node):
                return None
        return node
```

### Lazy Import Extraction

```python
class ImportExtractor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []

    def visit_ImportFrom(self, node):
        # from X import Y, Z
        for alias in node.names:
            self.imports.append((node.module, alias.name))

    def visit_Import(self, node):
        # import X  (harder to track what's used)
        for alias in node.names:
            self.imports.append((alias.name, None))
```

### User Module Detection

We only track user-defined modules, not stdlib or third-party:

```python
def is_user_module(module_name):
    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, '__file__'):
            return False
        path = Path(module.__file__)
        # Not in site-packages or stdlib
        return 'site-packages' not in str(path) and 'lib/python' not in str(path)
    except ImportError:
        return False
```

---

## Performance Considerations

### Hashing Cost

| Operation             | Time per Function |
| --------------------- | ----------------- |
| `inspect.getsource()` | ~0.1 ms           |
| `ast.parse()`         | ~0.05 ms          |
| Normalize AST         | ~0.02 ms          |
| `ast.dump()`          | ~0.1 ms           |
| xxhash                | ~0.001 ms         |
| **Total**             | **~0.3 ms**       |

For 100 stages with ~3 helper functions each: ~120 ms total. **Negligible.**

### When Hashing Occurs

Hashing happens at **pipeline load time** (when `pivot run` starts):

1. Import pipeline module
2. Collect all `@stage` decorated functions
3. Hash each function + its dependencies
4. Compare to stored hashes in lock files
5. Build list of stages to run

This is a one-time cost per run, not per-stage.

---

## Open Questions

We'd appreciate feedback on these questions:

### 1. Lazy Import Convention

Is requiring lazy imports for cross-module dependencies acceptable?

**Pro:** Makes dependencies explicit, AST can track them  
**Con:** Some developers strongly prefer top-level imports

**Alternative:** Could we make `code_deps` less error-prone somehow?

### 2. Global Variables

Should we attempt to track global variables used in functions?

```python
THRESHOLD = 0.5

@stage(...)
def train(...):
    if score > THRESHOLD:  # Uses global
        ...
```

**Option A:** Don't track, document as limitation  
**Option B:** Parse AST for `Name` nodes, resolve to module globals, hash their values

Option B is complex and might have false positives.

### 3. Package Version Tracking

Should we add optional package version tracking?

```python
@stage(..., package_deps=["pandas", "scikit-learn"])
def train(...):
    ...
```

This would hash the package versions and trigger re-run if they change.

**Pro:** Catches subtle behavior changes from package updates  
**Con:** Might be too sensitive (minor version bumps rarely break things)

### 4. What About Methods?

If a stage function calls `self.helper()` or `obj.process()`, we can't easily resolve what function that is statically.

**Current approach:** Don't track method calls, document as limitation.

**Alternative:** Track the class definition if it's in the same module?

### 5. Decorator Changes

If a helper function has decorators, should decorator changes trigger re-run?

```python
@cache  # If this decorator's implementation changes...
def helper(x):
    return expensive_computation(x)
```

**Current approach:** Don't track decorators on called functions.

---

## Appendix: Comparison with Other Tools

### DVC

DVC tracks:

- `cmd` string (the shell command)
- Input/output file hashes
- Parameter values

DVC does NOT track:

- Python function code (stages are shell commands)
- Dependencies between Python functions

### Kedro

Kedro tracks:

- Node function identity (but not code changes)
- Input/output datasets

Kedro does NOT have:

- Built-in incremental execution based on code changes

### Hamilton

Hamilton tracks:

- Function DAG based on type annotations

Hamilton does NOT have:

- Persistent caching between runs
- Code change detection

---

## Conclusion

Our proposed approach (transitive hashing + lazy import tracking) provides a good balance of:

- **Correctness**: Catches most common cases of code changes
- **Simplicity**: No complex static analysis
- **Explicitness**: Dependencies are visible in code
- **Performance**: Minimal overhead (~100ms for large pipelines)

The main tradeoff is requiring a coding convention (lazy imports for user utilities), which we believe is acceptable given the benefits.

**We welcome feedback on this design before implementation.**
