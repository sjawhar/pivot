---
tags: [python, typing, protocols, generics]
category: gotcha
module: type-system
symptoms: ["invariant list value type", "invariant dictionary value type", "Consider switching from list to Sequence"]
---

# Python List and Dict Invariance with Protocols

## Problem

A mutable container does not become covariant merely because its members satisfy
a protocol:

```python
class Descriptor(Protocol):
    ...

file_descriptors: list[FileDescriptor]
all_descriptors: list[Descriptor] = file_descriptors  # type error
```

The assignment would let callers append another `Descriptor` implementation to
a list that promises only `FileDescriptor` values.

## Solution

For read-only consumers, use covariant container interfaces:

```python
from collections.abc import Mapping, Sequence


def inspect_all(items: Sequence[Descriptor], by_name: Mapping[str, Descriptor]) -> None:
    ...
```

For mutation, keep a consistent concrete element type across the interface:

```python
descriptors: list[Descriptor] = []
descriptors.append(FileDescriptor())
```

## Key Insight

`list` and `dict` are invariant because mutation must remain type safe.
`Sequence` and `Mapping` are covariant read-only views. Choose a read-only
interface whenever the callee does not mutate the collection.
