# Code Fingerprinting

Pivot automatically detects when your Python code changes using a combination of AST hashing and `getclosurevars()` analysis.

## How It Works

```python
def helper(x):
    return x * 2  # Change this...

def process():
    return helper(load_data())  # ...and Pivot knows to re-run!
```

Pivot's fingerprinting system:

1. **Parses your function** into an Abstract Syntax Tree (AST)
2. **Discovers dependencies** using `inspect.getclosurevars()`
3. **Recursively fingerprints** all transitive dependencies
4. **Hashes the result** for fast comparison

## What Gets Tracked

| Pattern | Tracked? | How |
|---------|----------|-----|
| Same-module helpers | Yes | `getclosurevars()` discovers reference |
| Module imports | Yes | AST scanning for attribute access |
| Class definitions | Yes | Full class AST hashed |
| StageParams subclasses | Yes | Type hint analysis + full class AST + Pydantic field defaults + `default_factory` functions |
| Custom loaders | Yes | Loader class AST hashed via type hint analysis |
| Global constants | Yes (value) | `getclosurevars()` + `repr()` |
| Closure variables | Yes (structure) | `getclosurevars()` captures binding |
| Default arguments | Yes | Captured in function signature |
| `functools.partial` | Yes | Bound args + underlying function |

**Note:** While class AST is hashed (capturing method bodies and structure), *runtime values* of class-level variables are not tracked unless they're Pydantic model field defaults.

For Pydantic models, both static defaults and `default_factory` functions are tracked:

- **Static defaults** (e.g., `threshold: float = 0.5`): The value is hashed via `repr()`
- **Factory functions** (e.g., `items: list[str] = Field(default_factory=list)`): The factory function's AST is hashed, so changes to the factory logic trigger re-runs

## Normalization

AST normalization ensures stable hashing:

- **Function names** → Normalized to `"func"`
- **Docstrings** → Removed
- **Whitespace** → Ignored (AST doesn't capture it)
- **Comments** → Ignored (AST doesn't capture them)

This means:

```python
# These two functions have DIFFERENT fingerprints
def process(x): return x + 1
def process(x): return x + 2

# These two have THE SAME fingerprint
def process(x):
    return x + 1

def process(x): return x + 1  # Same logic
```

## Transitive Dependencies

Pivot tracks the entire dependency chain:

```python
def level_3(): return 1
def level_2(): return level_3() + 1
def level_1(): return level_2() + 1

def main():
    return level_1()
```

If `level_3()` changes, Pivot detects it and re-runs `main`.

## Import Patterns

Both import styles are supported:

```python
# Direct import
from helpers import process_data
# Tracked via getclosurevars()

# Module import (Google style)
import helpers
result = helpers.process_data(x)
# Tracked via AST scanning for module.attr patterns
```

## Known Limitations

Some patterns cannot be tracked automatically:

### Lazy Imports

```python
def process():
    from helpers import process_data  # Inside function!
    return process_data()
```

**Workaround:** Move imports to module level.

### Instance Method Calls

```python
processor = DataProcessor()

def process():
    return processor.transform(data)  # Method call
```

**Workaround:** Use function wrappers or explicit versioning.

### Lambda Functions

```python
# Lambda fingerprints are non-deterministic!
process = lambda x: x + 1  # Falls back to id(func), changes every session
```

Lambdas without accessible source code fall back to `id(func)`, which changes between interpreter sessions. This causes unnecessary re-runs.

**Workaround:** Always use named functions for stages.

### Dynamic Patterns

```python
func_name = "process"
func = getattr(module, func_name)  # Dynamic lookup
result = eval("process(data)")     # String execution
```

**Workaround:** Not supported; restructure code.

### Runtime-Computed Values

Global constants (primitives like `int`, `float`, `str`, `bool`, `bytes`, `None`) ARE tracked via `repr()`. However, values computed at runtime or by complex expressions are not:

```python
# TRACKED - primitive constants captured via repr()
THRESHOLD = 0.5  # Changing this value IS detected
DEBUG = True     # Changing this IS detected

# NOT TRACKED - runtime-computed values
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))  # Dynamic
CONFIG = load_config()  # Function call result
PATHS = [p for p in Path(".").glob("*.csv")]  # Computed at import
```

The fingerprinting system uses `getclosurevars()` to capture referenced globals, then stores `repr(value)` for primitives. This works for literal constants but cannot track values that change based on environment or runtime state.

**Workaround:** Use stage parameters for configurable values:

```python
# stages.py
from pivot.stage_def import StageParams


class ProcessParams(StageParams):
    threshold: float = 0.5


def process(params: ProcessParams):
    if value > params.threshold:  # Changes tracked!
        pass
```

```yaml
# pivot.yaml
stages:
  process:
    python: stages.process
    params:
      threshold: 0.5
```

## Why AST + getclosurevars?

Other approaches were tested and rejected:

| Approach | Problem |
|----------|---------|
| **Bytecode** | `co_code` doesn't include constants |
| **cloudpickle** | Pickles functions by reference, not content |
| **Comprehensive hashing** | Too slow, over-sensitive |

The AST + getclosurevars combination provides:

- **Accuracy** - Detects real code changes
- **Performance** - Fast hashing with xxhash64
- **Transitive tracking** - Follows dependency chains
- **Stability** - Ignores formatting changes

## Performance: Caching

Fingerprinting uses several `WeakKeyDictionary` caches to avoid repeated expensive operations:

| Cache | What it stores | Why it matters |
|-------|---------------|----------------|
| `_hash_function_ast_cache` | AST hash per function | Avoids re-parsing the same function |
| `_getclosurevars_cache` | Closure vars per function | `getclosurevars()` costs ~0.5ms per call |
| `_is_user_code_cache` | Boolean per object | Called 10K+ times, mostly repeats |
| `_get_type_hints_cache` | Type hints per function | `get_type_hints()` is expensive |
| `_module_attr_cache` | Module attribute usage per function | Avoids re-parsing AST for `module.attr` patterns |

These caches use weak references, so entries are automatically cleaned up when functions are garbage collected. The caches are not thread-safe, but fingerprinting runs single-threaded per process (multiprocessing uses separate memory spaces).

## Manifest Key Prefixes

The fingerprint manifest uses prefixed keys to distinguish different types of tracked items:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `self:` | The stage function itself | `self:` |
| `func:` | Helper functions (user code callables) | `func:process_data` |
| `class:` | Class definitions | `class:MyModel` |
| `const:` | Global constants (primitives via `repr()`) | `const:THRESHOLD` |
| `mod:` | Module attribute access patterns | `mod:utils.helper` |
| `partial:` | `functools.partial` bound arguments | `partial:0`, `partial:kwarg_name` |
| `loader:` | Loader class methods and config | `loader:load`, `loader:save`, `loader:config` |
| `pydantic:` | Pydantic model field defaults | `pydantic:field_name` |
| `builtin:` | Built-in types (for deterministic hashing) | `builtin:list` |

## Comparison with Other Tools

| Tool | Code Change Detection |
|------|----------------------|
| **DVC** | Hashes shell command string only |
| **Kedro** | No incremental execution based on code |
| **Hamilton** | No persistent caching between runs |
| **Prefect** | Caches by inputs, not code changes |
| **Dagster** | Hashes op config, not code |
| **Pivot** | AST + getclosurevars with transitive tracking |

## Debugging Fingerprints

Use explain mode to see what changed:

```bash
$ pivot explain train

Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Code changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4
      File: src/utils.py:15
```

## Manual Invalidation

If automatic detection fails, manually invalidate:

```bash
# Force re-run of specific stage
pivot run train --force
```

## Further Reading

- [Test Suite](https://github.com/sjawhar/pivot/tree/main/tests/fingerprint) - Comprehensive behavior tests
- [Architecture Overview](overview.md) - System design and other components
