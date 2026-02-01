---
tags: [python, typing, basedpyright, refactoring]
category: implementation
module: loaders, stage_def
symptoms: ["Cannot access attribute for class", "reportAttributeAccessIssue"]
---

# Type Narrowing with isinstance for Mixed Base Classes

## Problem

During a refactor splitting `Loader[T]` into separate `Reader[R]` and `Writer[W]` base classes, some code paths needed to call methods only available on the full `Loader` class (which implements both reading and writing).

For example, `FuncDepSpec.loader` was typed as `Reader[Any]` because dependencies are read. However, when an `IncrementalOut` is used as an input dependency, it provides a full `Loader` (not just a `Reader`), and the code needs to call `.empty()` which only exists on `Loader`:

```python
# FuncDepSpec.loader is typed as Reader[Any]
# But .empty() only exists on Loader
return spec.loader.empty()  # Type error: Cannot access attribute "empty" for class "Reader[Any]"
```

## Solution

Use `isinstance` checks to narrow the type from the broader interface to the specific class that has the method:

```python
if not spec.creates_dep_edge and not full_path.exists():
    # IncrementalOut as input: file doesn't exist yet (first run)
    # IncrementalOut provides a full Loader (not just Reader), so narrow the type
    if isinstance(spec.loader, loaders.Loader):
        return spec.loader.empty()
    raise RuntimeError(f"Loader for '{name}' does not support empty() for missing files")
```

This pattern:
1. Preserves the broad type in the interface (`Reader[Any]`)
2. Narrows to the specific type only where the extra capability is needed
3. Provides a clear error path if the assumption is violated at runtime

## Key Insight

When splitting a broad interface into narrower ones, some code paths legitimately need the full interface. Rather than:
- Widening the type everywhere (loses type safety)
- Adding the method to the narrow interface (pollutes the interface)

Use runtime `isinstance` checks to narrow the type locally. This documents the assumption explicitly and fails clearly if violated.

This is especially useful when the type hierarchy has constraints (like dataclass limitations preventing traditional inheritance) that force you to use broader union types in signatures.
