---
tags: [python, typing, generics, protocol]
category: gotcha
module: registry
symptoms: ["dict[str, BaseOut] cannot be assigned to dict[str, Out[Any]]", "Type parameter is invariant", "Consider switching from dict to Mapping"]
---

# Dict Invariance vs Protocol Covariance in Python Type System

## Problem

When refactoring to use a Protocol (`BaseOut`) instead of a concrete class (`Out[Any]`), type errors appeared when passing dictionaries:

```
Argument of type "dict[str, BaseOut]" cannot be assigned to parameter
"return_out_specs" of type "dict[str, Out[Any]]"
  "dict[str, BaseOut]" is not assignable to "dict[str, Out[Any]]"
    Type parameter "_VT@dict" is invariant, but "BaseOut" is not the same as "Out[Any]"
    Consider switching from "dict" to "Mapping" which is covariant in the value type
```

The issue arose when functions expected `dict[str, Out[Any]]` but were passed `dict[str, BaseOut]` after extraction functions were updated to return the more general Protocol type.

## Solution

Two approaches, depending on whether the function needs to mutate the dict:

1. **Use `Mapping` for read-only parameters** (covariant in value type):
   ```python
   def _validate_incremental_out_matching(
       return_out_specs: Mapping[str, BaseOut],  # Accepts dict[str, Out] or dict[str, BaseOut]
       ...
   ) -> None:
   ```

2. **Use `dict[str, BaseOut]` for parameters that need mutation**:
   ```python
   def process_specs(specs: dict[str, BaseOut]) -> None:
       specs["new_key"] = new_spec  # Can add BaseOut values
   ```

For list parameters, use `Sequence` (covariant) instead of `list` (invariant) when read-only access is sufficient.

## Key Insight

Python's generic collections have different variance properties:

| Mutable | Immutable | Variance |
|---------|-----------|----------|
| `dict[K, V]` | `Mapping[K, V]` | Key: invariant, Value: **covariant in Mapping** |
| `list[T]` | `Sequence[T]` | **Covariant in Sequence** |
| `set[T]` | `AbstractSet[T]` | **Covariant in AbstractSet** |

When introducing a Protocol to unify related types (`Out`, `DirectoryOut`, `IncrementalOut` all implementing `BaseOut`), you must also update function signatures to use covariant collection types (`Mapping`, `Sequence`) for parameters that don't need mutation. Otherwise, the type system will reject calls even though the Protocol relationship is correct.

This is especially important during large refactors that change return types of extraction functions - every downstream consumer of those dictionaries/lists must be audited.
