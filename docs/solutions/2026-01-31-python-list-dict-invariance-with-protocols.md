---
tags: [python, typing, protocols, generics]
category: gotcha
module: outputs
symptoms: ["Type parameter '_T@list' is invariant", "list[Subclass] not assignable to list[BaseClass]", "Consider switching from 'list' to 'Sequence'"]
---

# Python List/Dict Invariance with Protocols

## Problem

When introducing a `BaseOut` Protocol as a common interface for `Out`, `DirectoryOut`, and `IncrementalOut`, type errors appeared at multiple call sites:

```python
# In RegistryStageInfo TypedDict
outs: list[Out[Any]]  # Contains Out instances

# In WorkerStageInfo TypedDict
outs: list[BaseOut]  # Expects BaseOut Protocol

# Error when passing RegistryStageInfo.outs to WorkerStageInfo:
# "list[Out[Any]]" is not assignable to "list[BaseOut]"
# Type parameter "_T@list" is invariant
```

Similarly with dicts:
```python
out_specs: dict[str, Out[Any]]  # Source
out_specs: dict[str, BaseOut]   # Target

# Error: "dict[str, Out[Any]]" is not assignable to "dict[str, BaseOut]"
# Type parameter "_VT@dict" is invariant
```

## Root Cause

Python's `list` and `dict` are **invariant** in their type parameters, not covariant. This means even if `Out` satisfies the `BaseOut` Protocol, `list[Out]` is not a subtype of `list[BaseOut]`.

The reason is type safety: a `list[BaseOut]` could have a `DirectoryOut` appended to it, but a `list[Out]` cannot safely accept a `DirectoryOut`.

## Solution

Two approaches work:

### 1. Use Covariant Types (Sequence/Mapping)

For read-only access, use covariant container types:
```python
from typing import Sequence, Mapping

outs: Sequence[BaseOut]  # Covariant - accepts list[Out]
out_specs: Mapping[str, BaseOut]  # Covariant in values
```

### 2. Unify the Types

If all code paths construct and consume the same concrete types, use a consistent type throughout:
```python
# Both TypedDicts use the same type
outs: list[BaseOut]
out_specs: dict[str, BaseOut]
```

This was the chosen solution - changing `RegistryStageInfo` to use `BaseOut` since it can contain any output type (`Out`, `DirectoryOut`, `IncrementalOut`).

## Key Insight

When introducing a base Protocol/ABC for a type hierarchy, audit all container usages. Lists and dicts of the old type won't automatically work where the new base type is expected. Either:
1. Switch to covariant containers (`Sequence`, `Mapping`) for read-only access
2. Update all container types to use the base type consistently

The type checker's error message "Consider switching from 'list' to 'Sequence'" is a helpful hint, but unifying the types may be cleaner when mutation is needed.
