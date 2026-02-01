---
tags: [python, introspection, fingerprinting]
category: implementation
module: fingerprint
---

# MRO-Based Method Override Detection

## Problem

When fingerprinting loader classes, we need to detect whether a method like `empty()` has been overridden by a subclass. The naive approach using identity comparison doesn't work reliably:

```python
# Naive approach - fails
if loader.empty != Loader.empty:  # Identity comparison
    fingerprint_method(loader.empty)
```

This fails because method objects are created fresh on each attribute access, so identity comparison is unreliable.

## Solution

Walk the Method Resolution Order (MRO) and check each class's `__dict__` directly:

```python
def is_method_overridden(instance, method_name: str, base_class: type) -> bool:
    """Check if method is overridden by any class between instance and base_class."""
    for cls in type(instance).__mro__:
        if cls is base_class:
            return False  # Reached base without finding override
        if method_name in cls.__dict__:
            return True  # Found override before reaching base
    return False

# Usage in fingerprinting:
if isinstance(loader, Loader):
    for cls in type(loader).__mro__:
        if cls is Loader:
            break
        if "empty" in cls.__dict__:
            manifest[f"loader:{name}:empty"] = hash_function_ast(loader.empty)
            break
```

## Key Insight

To detect method overrides in Python:

1. **Don't use identity comparison** - Method objects are recreated on access
2. **Don't use `hasattr`** - It returns True for inherited methods too
3. **Use `cls.__dict__`** - Only contains methods defined directly on that class
4. **Walk the MRO** - Check each class in inheritance order until reaching the base

The MRO traversal ensures you detect overrides at any level of the inheritance hierarchy, not just the immediate subclass.
