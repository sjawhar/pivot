---
tags: [python, typing, basedpyright, refactoring]
category: implementation
module: loaders
symptoms: ["Cannot access attribute for class", "reportAttributeAccessIssue"]
---

# Type Narrowing with isinstance for Mixed Base Classes

## Problem

A value typed as a read-only or write-only interface cannot safely call methods
that exist only on a full `Loader`. For example, incremental output setup needs
`empty()` for its first run, while a general format value may be a `Reader`, a
`Writer`, or a `Loader`.

```python
format: Reader[object] | Writer[object] | Loader[object, object]
format.empty()  # type error: Reader and Writer do not define empty
```

## Solution

Use a runtime check to narrow to the interface that carries the required
method:

```python
if not isinstance(format, loaders.Loader):
    raise StageDefinitionError("incremental outputs require a Loader")

initial_value = format.empty()
```

When the first-run value must be meaningful, validate that the concrete loader
overrides a base implementation that only raises.

## Key Insight

Keep broad interfaces at boundaries and narrow locally where a specific
capability is required. This retains type safety, documents the assumption, and
produces a clear boundary error rather than a late attribute failure.
