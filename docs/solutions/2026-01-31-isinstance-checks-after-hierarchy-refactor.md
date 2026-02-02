---
tags: [python, refactoring, isinstance, type-hierarchy]
category: gotcha
module: stage_def, outputs
symptoms: ["IncrementalOut not recognized", "DirectoryOut not detected", "single output spec returns None"]
---

# Update isinstance Checks When Splitting Inheritance Hierarchies

## Problem

When refactoring `Loader[T]` into separate `Reader[R]` and `Writer[W]` base classes, `IncrementalOut` and `DirectoryOut` were changed from `Out` subclasses to standalone classes implementing a `BaseOut` protocol.

After the refactor, `get_single_output_spec_from_return()` stopped recognizing `IncrementalOut` and `DirectoryOut` as valid output specs because it only checked:

```python
if isinstance(metadata, outputs.Out):
    return metadata
```

## Solution

Update the isinstance check to include all concrete output types:

```python
if isinstance(metadata, (outputs.Out, outputs.IncrementalOut, outputs.DirectoryOut)):
    return metadata
```

Also update the return type annotation from `Out[Any] | None` to `BaseOut | None` to reflect the broader type.

## Key Insight

When refactoring a class hierarchy (especially when converting subclasses to siblings or protocol implementers), search the entire codebase for `isinstance` checks against the old base class. Each one needs review:

1. **Find all isinstance checks**: `grep -r "isinstance.*OldBaseClass"`
2. **Evaluate each usage**: Does it need to match the old subclasses that are now siblings?
3. **Consider alternatives**:
   - Add all concrete types to the isinstance tuple
   - Use a Protocol with `@runtime_checkable` and check against that
   - Use a Union type alias for the concrete types

This is a common gotcha because isinstance checks silently return `False` for the old subclasses - they don't raise errors, they just stop matching.
