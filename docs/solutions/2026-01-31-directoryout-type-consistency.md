---
tags: [python, generics, type-system, outputs]
category: design
module: outputs
---

# DirectoryOut Generic Type Mismatch

## Problem

`DirectoryOut[T]` had a type inconsistency: it holds a loader that operates on individual `T` values, but the stage actually returns `dict[str, T]`. This meant:
- The loader field type was `Loader[T]`
- But the stage return type annotation was `dict[str, T]`

This was a "type lie" where the generic parameter meant different things in different contexts.

## Options Considered

1. **Don't inherit from `Out`** - Simple but duplicates `path`, `cache` fields
2. **Inherit as `Out[T]`** - Confusing since stage returns `dict[str, T]`, not `T`
3. **Two-level typing** with `item_loader: Writer[T]` - Awkward with unused inherited `loader`
4. **Override `loader` with different type** - Violates Liskov Substitution Principle
5. **Common base class** - Extract shared fields, each subclass defines its own `loader` type

## Solution

Extract a common base class without the `loader` field:

```python
class BaseOut(Generic[T]):
    path: PathType
    cache: bool = True

class Out(BaseOut[T]):
    loader: Writer[T]

class DirectoryOut(BaseOut[dict[str, T]]):
    loader: Writer[T]  # T is item type, dict[str, T] is return type
```

This makes the types explicit:
- `BaseOut[dict[str, T]]` correctly says "the stage returns `dict[str, T]`"
- `loader: Writer[T]` correctly says "individual items are serialized as `T`"

## Key Insight

When a generic class has attributes that operate on a *component* of the generic type parameter (like `DirectoryOut` having a loader for items, not the whole dict), the cleaner solution is to use a base class that captures the "outer" type and let the subclass add the "inner" type attribute. This avoids type lies and Liskov violations.

More generally: if you find yourself wanting a subclass attribute to have an incompatible type with the parent, that's a signal the inheritance hierarchy needs restructuring, not a targeted `# type: ignore`.
