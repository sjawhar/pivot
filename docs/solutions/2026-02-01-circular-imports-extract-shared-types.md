---
tags: [python, imports, architecture]
category: design
module: types
symptoms: ["ImportError: cannot import name", "circular import detected", "partially initialized module"]
---

# Circular Imports: Extract Shared Types to a Separate Module

## Problem

When two modules import from each other, Python raises an `ImportError`:

```python
# module_a.py
from module_b import SomeType

class ResultA:
    data: SomeType

# module_b.py
from module_a import ResultA  # ImportError: cannot import name 'ResultA'

class SomeType:
    result: ResultA
```

This commonly happens when:
1. A registry module imports stage types, and stage modules import from the registry
2. An executor module imports result types defined alongside the functions that produce them
3. Two modules have legitimate bidirectional data flow relationships

The error message varies by Python version and import timing:
- `ImportError: cannot import name 'X' from partially initialized module 'Y'`
- `AttributeError: module 'Y' has no attribute 'X'`

## Solution

Extract shared type definitions to a dedicated module that has no dependencies on the importing modules:

```python
# types.py - no imports from module_a or module_b
class SomeType:
    result: "ResultA"  # Forward reference as string

class ResultA:
    data: SomeType

# module_a.py
from types import ResultA, SomeType
# ... uses ResultA

# module_b.py
from types import ResultA, SomeType
# ... uses SomeType
```

In Pivot, this is `pivot/types.py`, which contains:
- TypedDicts (StageResult, LockData, HashInfo, etc.)
- Enums (StageStatus, OnError, ChangeType, etc.)
- Type aliases (OutputHash, StageFunc, etc.)
- TypeGuard functions for type narrowing

The key property: `types.py` imports only from the standard library and typing modules, never from other Pivot modules (except under `TYPE_CHECKING` for type hints only).

```python
# pivot/types.py
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pivot.run_history import RunCacheEntry  # Only for type hints

class DeferredWrites(TypedDict):
    run_cache_entry: "RunCacheEntry"  # String annotation avoids runtime import
```

## Key Insight

Circular imports are a symptom of tangled dependencies. The fix isn't import tricks (lazy imports, inline imports)—it's restructuring the dependency graph.

The pattern:
1. **Identify the shared types** that create the cycle
2. **Extract them to a leaf module** with no internal dependencies
3. **Both original modules import from the leaf**

This transforms a cycle (`A -> B -> A`) into a tree (`A -> types <- B`).

For type hints that would create import cycles, use `TYPE_CHECKING` blocks:
```python
if TYPE_CHECKING:
    from heavy_module import ExpensiveClass

def process(item: "ExpensiveClass") -> None:  # String annotation
    ...
```

The string annotation defers resolution to type-checking time, avoiding runtime import.
