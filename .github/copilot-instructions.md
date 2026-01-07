# Copilot Instructions for Pivot

## Python 3.13+ Type Syntax

This project uses Python 3.13+ and leverages the **constructor syntax for typed empty collections**, which is valid and preferred in this codebase:

```python
# PREFERRED - Python 3.13+ constructor syntax (what this project uses)
items = list[str]()           # Creates empty list[str]
mapping = dict[str, int]()    # Creates empty dict[str, int]
unique = set[int]()           # Creates empty set[int]
```

**Do NOT flag `list[T]()`, `dict[K, V]()`, or `set[T]()` as errors** - these are intentional and preferred in this codebase for clarity and type inference.

## Import Style

This project follows Google Python Style Guide - import modules, not functions:

```python
# CORRECT
import pathlib
import pydantic
from pivot import fingerprint

path = pathlib.Path("/some/path")
fp = fingerprint.get_stage_fingerprint(func)

# INCORRECT
from pathlib import Path
from pivot.fingerprint import get_stage_fingerprint
```

The exception to this rule is type hints in `TYPE_CHECKING` blocks, where you may import types directly.
