---
tags: [python, typing, generics]
category: gotcha
module: registry
symptoms: ["dict[str, BaseClass] cannot be assigned to dict[str, SubClass]", "Type parameter is invariant", "Consider switching from dict to Mapping"]
---

# Dict Invariance Causes Type Errors with Class Hierarchies

## Problem

When refactoring a class hierarchy (e.g., splitting `Loader[T]` into `Reader` and `Writer` base classes), type errors appear when passing dicts between functions:

```
Argument of type "dict[str, BaseOut]" cannot be assigned to parameter of type "dict[str, Out[Any]]"
  "dict[str, BaseOut]" is not assignable to "dict[str, Out[Any]]"
    Type parameter "_VT@dict" is invariant, but "BaseOut" is not the same as "Out[Any]"
```

This happens even though `Out[Any]` is a subclass of `BaseOut`.

## Solution

Two options:

1. **Use `Mapping` for read-only access** (covariant in value type):
   ```python
   def process_outputs(specs: Mapping[str, BaseOut]) -> None:
       for name, spec in specs.items():
           ...
   ```

2. **Align types across the interface** - ensure all functions in a call chain use the same specific type:
   ```python
   # If worker needs Out[Any] specifically, store Out[Any] not BaseOut
   out_specs: dict[str, Out[Any]]  # Not dict[str, BaseOut]
   ```

## Key Insight

Python's `dict` is **invariant** in both key and value types. This means:
- `dict[str, Child]` is NOT a subtype of `dict[str, Parent]`
- `dict[str, Parent]` is NOT a subtype of `dict[str, Child]`

This is because dicts are mutable - if you could assign a `dict[str, Child]` to a `dict[str, Parent]` variable, you could then insert a different `Parent` subclass, violating type safety.

`Mapping` is read-only and therefore covariant, allowing the intuitive subtype relationship. Use `Mapping` for function parameters that only read from the dict; use `dict` for return types or when mutation is needed.

When designing class hierarchies with generics, plan the type annotations across the entire call chain upfront to avoid cascading invariance errors during refactoring.
