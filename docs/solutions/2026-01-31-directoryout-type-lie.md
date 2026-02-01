---
tags: [python, generics, type-safety, outputs]
category: gotcha
module: outputs
symptoms: ["type mismatch between inheritance and usage", "loader type doesn't match Out generic"]
---

# DirectoryOut Inheritance Creates Type Lie

## Problem

`DirectoryOut[T]` inherits from `Out[dict[str, T]]`, which implies its `loader` field has type `Loader[dict[str, T]]`. However, the actual implementation uses the loader to serialize individual `T` values, not `dict[str, T]`.

Example demonstrating the inconsistency:
```python
# DirectoryOut[TaskMetrics] extends Out[dict[str, TaskMetrics]]
# So loader should be Loader[dict[str, TaskMetrics]]
# But YAML() here is Loader[TaskMetrics]:
DirectoryOut("metrics/task_results/", YAML())

# The code then iterates over the dict and calls loader.save() on each value (T),
# not on the dict itself (dict[str, T])
for key, item_value in data.items():
    write_ops.append((full_path, item_value, spec.loader))
```

## Solution

When refactoring to Reader/Writer split, `DirectoryOut` cannot simply inherit from `Out`. It needs its own `loader: Writer[T]` field (not `Writer[dict[str, T]]`).

Options:
1. **Override the loader field** - `DirectoryOut` declares its own `loader: Writer[T]`
2. **Don't inherit from Out** - Make `DirectoryOut` standalone with similar interface
3. **Composition over inheritance** - `DirectoryOut` contains behavior, doesn't extend `Out`

## Key Insight

When a generic subclass uses its type parameter differently than the parent (T vs dict[str, T]), the inheritance relationship creates a "type lie" - the static types don't match runtime behavior. This often indicates the inheritance is modeling the wrong relationship.

**The real relationship:** `DirectoryOut` is not an `Out` that happens to produce `dict[str, T]`. It's a different abstraction (a factory for multiple Outs) that shares some surface-level interface.

**General principle:** If a generic subclass transforms the type parameter in its inheritance clause (e.g., `class Child[T](Parent[SomeWrapper[T]])`), verify that all inherited fields still make sense with the transformed type.
