---
tags: [python, dataclass, inheritance, typing]
category: gotcha
module: outputs
symptoms: ["TypeError: non-default argument follows default argument", "dataclass field ordering error"]
---

# Dataclass Inheritance: Non-Default Fields Cannot Follow Fields with Defaults

## Problem

When refactoring the output class hierarchy to extract a common `BaseOut` base class, the plan was:

```python
@dataclasses.dataclass(frozen=True)
class BaseOut[T]:
    path: PathType
    cache: bool = True  # Has a default

@dataclasses.dataclass(frozen=True)
class Out[T](BaseOut[T]):
    loader: Writer[T]  # No default - ERROR!
```

This fails because Python dataclasses collect fields from the entire class hierarchy, and **non-default fields cannot appear after fields with defaults**. Since `BaseOut` has `cache: bool = True`, any subclass adding a field without a default (`loader`) violates the ordering rule.

This is a fundamental limitation of dataclass inheritance, not a bug.

## Solution

Make the child classes standalone (no inheritance from `BaseOut`), accepting the small amount of field duplication:

```python
@dataclasses.dataclass(frozen=True)
class Out[W]:
    path: PathType
    loader: Writer[W]
    cache: bool = True

@dataclasses.dataclass(frozen=True)
class DirectoryOut[T]:
    path: str  # Duplicated field
    loader: Writer[T]
    cache: bool = True  # Duplicated field

@dataclasses.dataclass(frozen=True)
class IncrementalOut[W, R = W]:
    path: PathType  # Duplicated field
    loader: Loader[W, R]
    cache: bool = True  # Duplicated field
```

## Key Insight

When designing dataclass hierarchies, consider field ordering upfront:

1. **All non-default fields must come before all default fields** across the entire inheritance chain
2. If a base class has fields with defaults, subclasses cannot add required fields
3. Workarounds:
   - Make all fields have defaults (use sentinel values or `field(default=MISSING)`)
   - Use composition instead of inheritance
   - Accept field duplication in standalone classes (often simplest)
   - Use `attrs` library which has more flexible field ordering via `kw_only`

The duplication tradeoff is often acceptable: three classes each with `path`, `loader`, `cache` is cleaner than complex inheritance gymnastics.
