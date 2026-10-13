---
tags: [python, typing, generics, protocol]
category: gotcha
module: type-system
symptoms: ["invariant dictionary value type", "Consider switching from dict to Mapping"]
---

# Dict Invariance vs Protocol Covariance in Python Type System

## Problem

A mutable dictionary is invariant in its value type. A function accepting
`dict[str, ConcreteSpec]` cannot accept `dict[str, SpecProtocol]`, even when
`ConcreteSpec` implements `SpecProtocol`.

```text
Argument of type "dict[str, SpecProtocol]" cannot be assigned to parameter
of type "dict[str, ConcreteSpec]"
```

This often appears after an extraction step broadens its result from a concrete
class to a structural protocol.

## Solution

Choose the container based on whether the callee mutates it:

1. Use a read-only `Mapping` for inputs the callee only reads:

   ```python
   def validate_specs(specs: Mapping[str, SpecProtocol]) -> None:
       ...
   ```

2. Use one consistent mutable dictionary type when the callee inserts or
   replaces values:

   ```python
   def extend_specs(specs: dict[str, SpecProtocol]) -> None:
       specs["derived"] = make_spec()
   ```

Use `Sequence` rather than `list` for read-only collections for the same
reason.

## Key Insight

`dict` and `list` are invariant because callers can mutate them. `Mapping` and
`Sequence` are covariant views. Select the narrowest container interface that
the callee needs rather than widening a concrete type only to satisfy an
assignment.
