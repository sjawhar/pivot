---
tags: [python, generics, type-system, outputs]
category: design
module: type-system
---

# Directory-Style Output Generic Type Consistency

## Problem

A directory-style result often returns `dict[str, T]` while its serializer
writes each individual `T`. Treating the serializer as though it handled the
whole dictionary hides that distinction and produces misleading type
annotations.

## Options Considered

1. Model the result as a plain `dict[str, T]` and keep its item writer separate.
2. Inherit from a file-output type and override the writer type.
3. Introduce a wrapper with explicit result and item types.

The inheritance option makes the parent contract misleading because the parent
writes a complete result while the child writes individual entries.

## Solution

Use explicit names for both levels of the data model:

```python
@dataclasses.dataclass(frozen=True)
class FileGroup[T]:
    path: str
    item_writer: Writer[T]


def write_group(group: FileGroup[T], values: dict[str, T]) -> None:
    for relative_path, value in values.items():
        group.item_writer.save(value, group.path / relative_path)
```

The type of the stage result is `dict[str, T]`; the type of the writer input is
`T`. Keeping them separate makes the serialization contract visible.

## Key Insight

When one attribute operates on a component of a result rather than the result
itself, encode both types directly. Do not force an inheritance relationship
whose generic parameter means different things at different levels.
