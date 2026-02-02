---
tags: [python, generics, typing, python-3.13]
category: design
module: loaders
---

# PEP 696 Type Parameter Defaults for Asymmetric Generics

## Problem

When designing a `Loader` class that can both read and write data, most loaders are symmetric (same type for read and write), but some are asymmetric (e.g., write a matplotlib `Figure`, read back as `np.ndarray`).

The naive approach requires two type parameters for all loaders:
```python
class Loader[W, R](Writer[W], Reader[R]): ...

# Every loader needs both params, even symmetric ones:
class CSV(Loader[DataFrame, DataFrame]): ...  # Verbose
class JSON(Loader[dict, dict]): ...           # Redundant
```

This is verbose for the common symmetric case.

## Solution

Use PEP 696 (Python 3.13+) type parameter defaults:

```python
class Loader[W, R = W](Writer[W], Reader[R]):
    """Bidirectional loader. R defaults to W for symmetric loaders."""
    def empty(self) -> R:
        raise NotImplementedError(f"{type(self).__name__} doesn't support empty()")
```

Usage becomes clean for both cases:

```python
# Symmetric - single type param uses default (R = W)
class CSV(Loader[DataFrame]):
    def save(self, data: DataFrame, path: Path) -> None: ...
    def load(self, path: Path) -> DataFrame: ...

# Asymmetric - explicit both types
class PngImage(Loader[Figure, np.ndarray]):
    """Write matplotlib Figure, read back as numpy array."""
    def save(self, data: Figure, path: Path) -> None: ...
    def load(self, path: Path) -> np.ndarray: ...
```

## Key Insight

PEP 696 type parameter defaults (`R = W`) provide a clean solution for the "usually the same, sometimes different" pattern in generic types. This avoids:

1. Verbose type aliases (`type SymmetricLoader[T] = Loader[T, T]`)
2. Runtime type inference hacks
3. Forcing all users to specify redundant type parameters

The syntax `class Foo[A, B = A]` works in Python 3.13+ and type checkers like basedpyright/pyright support it fully.

**Parameter ordering matters:** Put the "usually defaulted" parameter second. Here, `W` (write type) comes first because the dataflow is save(W) -> file -> load() -> R, and most users think about what they're writing first.
