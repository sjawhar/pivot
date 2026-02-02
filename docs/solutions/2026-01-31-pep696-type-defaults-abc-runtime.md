---
tags: [python, typing, generics, abc]
category: gotcha
module: loaders
symptoms: ["TypeError: Some type variables (W) are not listed in Generic[T]", "runtime error with type parameter defaults"]
---

# PEP 696 Type Parameter Defaults Don't Work with ABC at Runtime

## Problem

When implementing a generic base class with type parameter defaults (PEP 696) that inherits from `abc.ABC`, subclasses using the default fail at runtime:

```python
@dataclasses.dataclass(frozen=True)
class Loader[W, R = W](Writer[W], Reader[R], abc.ABC):
    """W is write type, R is read type. Defaults to symmetric (R = W)."""
    ...

# This FAILS at runtime:
class CSV[T](Loader[T]):  # Uses default R = T
    ...

# Error:
# TypeError: Some type variables (W) are not listed in Generic[T]
```

The error occurs because Python's ABC machinery (`abc.ABCMeta.__new__`) processes type variables before the default is applied, seeing `W` and `R` as separate unbound variables.

## Solution

Explicitly provide both type parameters even when they're the same:

```python
# This WORKS:
class CSV[T](Loader[T, T]):  # Explicitly symmetric
    ...

class Text(Loader[str, str]):  # Concrete symmetric
    ...
```

## Key Insight

**PEP 696 type parameter defaults are a static typing feature only** - they work for type checkers but not for runtime introspection by metaclasses like `abc.ABCMeta`. When combining generics with ABCs, always explicitly specify all type parameters in subclass definitions, even if the values are redundant.

This affects any pattern where:
1. A base class uses `class Foo[A, B = A]` with defaults
2. The base class inherits from `abc.ABC` (directly or indirectly)
3. Subclasses try to use the default with `class Bar(Foo[X])`
