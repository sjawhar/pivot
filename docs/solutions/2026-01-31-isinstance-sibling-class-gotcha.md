---
tags: [python, refactoring, inheritance, isinstance]
category: gotcha
module: outputs
symptoms: ["tests failing after refactoring class hierarchy", "isinstance check silently missing types", "function returns None unexpectedly"]
---

# isinstance Checks Break When Refactoring Subclass to Sibling

## Problem

When refactoring a class hierarchy from subclass relationships to sibling relationships, existing `isinstance` checks silently fail to match the new sibling classes.

In this case, the Loader hierarchy was refactored from:
```
Before: Out <- IncrementalOut, DirectoryOut (subclasses)
After:  BaseOut <- Out, IncrementalOut, DirectoryOut (siblings)
```

Code using `isinstance(obj, Out)` to detect any output spec stopped matching `IncrementalOut` and `DirectoryOut` after the refactor. The bug was silent - no exceptions, just incorrect behavior (returning `None` instead of the expected spec).

## Solution

After refactoring to sibling classes with a common base:
1. Search for all `isinstance` checks against the old parent class
2. Update to either:
   - Check against the new common base: `isinstance(obj, BaseOut)`
   - Check for all siblings: `isinstance(obj, (Out, IncrementalOut, DirectoryOut))`
3. Run tests to verify - these bugs often only manifest in integration tests

In `get_single_output_spec_from_return`:
```python
# Before (broken after refactor)
if isinstance(metadata, outputs.Out):
    return cast("outputs.Out[Any]", metadata)

# After (handles all output spec types)
if isinstance(metadata, outputs.BaseOut):
    return metadata
```

## Key Insight

When changing a class hierarchy from inheritance to composition/siblings, `isinstance` checks are silent failure points. Unlike method calls (which raise `AttributeError` if missing), isinstance returns `False` for non-matching types without any indication that it used to match.

Mitigation strategies:
- Search for `isinstance(*, OldParent)` patterns before refactoring
- Add explicit tests for each concrete type at API boundaries
- Consider using structural typing (Protocols) instead of isinstance when the exact hierarchy might change
