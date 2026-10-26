---
tags: [python, refactoring, isinstance, type-hierarchy]
category: gotcha
module: type-system
symptoms: ["metadata not recognized", "single result omitted"]
---

# Update isinstance Checks When Splitting Inheritance Hierarchies

## Problem

After moving related metadata classes from a parent-child hierarchy to sibling
classes, an `isinstance` check against the former parent silently stops matching
the siblings:

```python
if isinstance(metadata, FileMetadata):
    return metadata
```

The failure is quiet: unmatched metadata is simply ignored.

## Solution

Review every check against the old parent and choose an explicit replacement:

```python
if isinstance(metadata, (FileMetadata, DirectoryMetadata)):
    return metadata
```

When the code only needs a shared runtime shape, a deliberately designed
runtime-checkable protocol can be clearer than a growing tuple of concrete
classes.

## Key Insight

Hierarchy refactors require a search for every `isinstance` check against the
changed parent. Test each resulting branch because a false result does not
raise an exception; it can silently remove data from the result.
