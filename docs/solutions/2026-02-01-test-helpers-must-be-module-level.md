---
tags: [python, testing, fingerprinting, closures]
category: gotcha
module: fingerprint
symptoms: ["fingerprint doesn't detect import changes", "test helper imports not tracked", "stage helper function changes not triggering re-runs"]
---

# Test Helpers Must Be Module-Level for Fingerprinting

## Problem

When helper functions are defined inline inside test functions, `inspect.getclosurevars()` cannot see their module-level imports. This causes fingerprinting to miss import dependencies, so changes to imported modules don't trigger stage re-runs.

```python
import math

def test_fingerprint_tracking():
    # BROKEN: math won't appear in getclosurevars() result
    def inline_helper():
        return math.pi * 2

    manifest = fingerprint.get_stage_fingerprint(inline_helper)
    # manifest won't contain math module - changes to math usage undetected!
```

The issue stems from how Python closures work. `getclosurevars()` returns:
- `globals`: Names from the function's `__globals__` that are referenced in the code
- `nonlocals`: Variables from enclosing scopes captured in `__closure__`

For an inline function, `math` is in the *test function's* globals, not the inline function's direct globals. The inline function's `__globals__` points to the module's global namespace, but `getclosurevars()` only reports names actually referenced by the function's bytecode - and `math` appears to be referenced through the outer scope.

## Solution

Define helper functions at module level with a `_helper_` prefix:

```python
import math

# Module level - getclosurevars() properly captures math reference
def _helper_uses_math():
    return math.pi * 2

def test_fingerprint_tracking():
    manifest = fingerprint.get_stage_fingerprint(_helper_uses_math)
    # manifest contains mod:math.pi - changes detected correctly
```

For test files, follow this pattern:

```python
# tests/test_my_feature.py

import pandas as pd
from pivot import fingerprint

# --- Module-level helper functions ---
# These must be at module level to properly capture imports in closures

def _helper_process_dataframe():
    """Helper that uses pandas."""
    return pd.DataFrame({"a": [1, 2, 3]})

def _helper_transform_data(df):
    """Helper that transforms data."""
    return df.dropna()

# --- Tests ---

def test_dataframe_processing():
    manifest = fingerprint.get_stage_fingerprint(_helper_process_dataframe)
    # pandas usage is now tracked in the fingerprint
    assert any("pd" in key or "pandas" in key for key in manifest)
```

## Key Insight

`getclosurevars()` inspects a function's direct references, not transitive ones through enclosing scopes. Inline functions inherit their module's `__globals__` dict, but the closure inspection only captures what the function's bytecode directly references.

For fingerprinting to work correctly:
1. Helper functions must be defined at module level
2. Imports they use must also be at module level (no lazy imports inside functions)
3. Use `_helper_` prefix to distinguish test helpers from actual test functions

This is why Pivot's CLAUDE.md mandates: "Stage functions, output TypedDicts, and custom loaders must be module-level" - the same principle applies to any code that needs accurate fingerprinting.
