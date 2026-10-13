---
tags: [python, dataclass, inheritance, typing]
category: gotcha
module: type-system
symptoms: ["TypeError: non-default argument follows default argument", "dataclass field ordering error"]
---

# Dataclass Inheritance: Non-Default Fields Cannot Follow Fields with Defaults

## Problem

A shared dataclass can make a subclass impossible to construct when the shared
class has a defaulted field and the subclass adds a required field:

```python
@dataclasses.dataclass(frozen=True)
class SharedSpec:
    cache: bool = True


@dataclasses.dataclass(frozen=True)
class FileSpec(SharedSpec):
    loader: Writer[object]
```

Python collects fields across the inheritance chain, so the required `loader`
follows the defaulted `cache` field. That ordering is invalid.

## Solution

Use standalone dataclasses when the variants need their own required fields:

```python
@dataclasses.dataclass(frozen=True)
class FileSpec:
    path: str
    loader: Writer[object]
    cache: bool = True


@dataclasses.dataclass(frozen=True)
class DirectorySpec:
    path: str
    item_loader: Writer[object]
    cache: bool = True
```

A small amount of field duplication is simpler than an inheritance hierarchy
that depends on field-order workarounds.

## Key Insight

Across a dataclass inheritance chain, every required field must precede every
defaulted field. When a subtype needs a required field after a shared default,
prefer composition or a standalone dataclass. Other options are a sentinel
default, keyword-only fields, or a library with different field-order rules.
