---
tags: [python, refactoring, inheritance, isinstance]
category: gotcha
module: type-system
symptoms: ["tests failing after refactoring class hierarchy", "isinstance check silently missing types", "function returns None unexpectedly"]
---

# isinstance Checks Break When a Subclass Becomes a Sibling

## Problem

Changing a class hierarchy can make an existing `isinstance` check silently
return `False` for a class that previously matched. This is especially easy to
miss when the check chooses optional metadata or a serialization strategy.

```python
# Before the hierarchy change
if isinstance(metadata, ParentMetadata):
    return metadata
```

## Solution

After the change, make the required cases explicit:

```python
if isinstance(metadata, (FileMetadata, DirectoryMetadata)):
    return metadata
```

If the code depends only on a stable set of attributes, use a
runtime-checkable protocol designed for that shared shape instead of relying
on inheritance.

## Key Insight

`isinstance` failures are silent. When changing inheritance relationships,
search for every parent-class check, decide which replacement cases are valid,
and add a behavioral test for each affected boundary.
