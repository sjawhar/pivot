---
tags: [python, typing, generics]
category: gotcha
module: type-system
symptoms: ["invariant dictionary value type", "Consider switching from dict to Mapping"]
---

# Dict Invariance Causes Type Errors with Class Hierarchies

## Problem

A dictionary of a subtype is not a dictionary of its parent type:

```text
dict[str, ChildSpec] is not assignable to dict[str, ParentSpec]
```

This holds even when every `ChildSpec` is a valid `ParentSpec`. The target
dictionary could otherwise accept a different parent subtype and corrupt the
source dictionary's invariant.

## Solution

For a read-only parameter, accept a `Mapping`:

```python
def process_specs(specs: Mapping[str, ParentSpec]) -> None:
    for name, spec in specs.items():
        ...
```

For a mutable interface, align the concrete dictionary types through the call
chain:

```python
specs: dict[str, ParentSpec] = {}
specs["file"] = ChildSpec()
```

## Key Insight

`dict` is invariant in its key and value types because it is mutable. `Mapping`
is read-only and covariant in its values. Use `Mapping` when mutation is not
part of the contract; otherwise choose one dictionary value type for every
producer and consumer.
