# Common Gotchas

Pitfalls and discoveries from Pivot development. Read this to avoid rediscovering these issues.

## Fingerprinting

### Test Helpers Must Be Module-Level

`getclosurevars()` doesn't see imports in inline closures:

```python
# Works - module level
def _helper_uses_math():
    return math.pi

# FAILS - inline in test
def test_it():
    def uses_math():  # math won't be in closure!
        return math.pi
```

### Single Underscore Functions ARE Tracked

Only dunders (`__name__`, `__init__`) are filtered from fingerprinting. Single underscore functions like `_helper()` ARE tracked.

### Lambda Fingerprinting Is Non-Deterministic

Lambdas without accessible source code fall back to `id(func)`, which changes every interpreter session. This causes unnecessary re-runs.

**Always use named functions in stage definitions:**

```python
# Bad - non-deterministic fingerprint
filter_func = lambda x: x > 0.5

# Good - stable AST-based fingerprint
def filter_positive(x: float) -> bool:
    return x > 0.5
```

## Multiprocessing

### loky Can't Pickle `mp.Queue()`

`multiprocessing.Queue()` can't be pickled for loky workers. Use manager queues:

```python
# Bad
queue = mp.Queue()

# Good
queue = mp.Manager().Queue()
```

### Cross-Process Tests Need File-Based State

Shared lists don't work across processes - each process gets a copy:

```python
# Bad - each process has its own copy
execution_log = list[str]()

def my_stage():
    execution_log.append("ran")  # Silently fails!

# Good - file-based logging
def my_stage():
    with open("log.txt", "a") as f:
        f.write("ran\n")
```

### Stage Functions Must Be Module-Level

Workers receive pickled functions. Lambdas, closures, and `__main__` definitions fail.

```python
# Good
def process_data(): ...

# Bad - closure captures variable
def make_stage(threshold):
    def process():
        if value > threshold:  # Captures threshold!
            ...
```

## Types

### Stage Functions and TypedDicts Must Be Module-Level

`get_type_hints()` needs importable `__module__` for type resolution.

```python
# Good - module level
class ProcessOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("out.csv", loaders.PathOnly())]

def process(...) -> ProcessOutputs:
    ...

# Bad - inline
def create_stage():
    class ProcessOutputs(TypedDict):  # Can't resolve type hints!
        ...
```

### AST Manipulation: Function Bodies Need Statements

Empty function bodies cause AST errors. Add `ast.Pass()` if needed:

```python
# When manipulating AST, ensure body has content
if not func_node.body:
    func_node.body = [ast.Pass()]
```

## Storage

### IncrementalOut Uses COPY Mode

`IncrementalOut` always copies from cache rather than hardlinking/symlinking. This is intentional - the stage modifies the file in-place, so hardlinks would corrupt the cache.

### Path Strategies in StateDB

- Use `resolve()` for hash keys (deduplication - `/foo/../bar` and `/bar` are same file)
- Use `normpath()` for generation keys (logical paths for user display)

### Atomic Writes: Track fd Closure

When using `mkstemp()` + rename for atomic writes, ensure the file descriptor is closed before rename:

```python
fd, tmp_path = tempfile.mkstemp()
try:
    os.write(fd, data)
finally:
    os.close(fd)  # Must close before rename!
os.rename(tmp_path, final_path)
```

## Dependencies

### Path Overlap Detection: Use pygtrie

String matching fails for path overlap detection (`data/` vs `data/file.csv`). Use pygtrie for prefix trees.

### LMDB for All State

Don't create new databases. Extend `StateDB` with prefixes:

```python
# Good - use prefixes
state_db.set("myfeature:key", value)

# Bad - new database
my_db = lmdb.open("my_state.lmdb")
```

## Configuration

### ruamel.yaml vs PyYAML

- **ruamel.yaml** - For editable config (preserves comments)
- **PyYAML** - For read-only (faster)

```python
# Editing config - preserve comments
import ruamel.yaml
yaml = ruamel.yaml.YAML()

# Read-only
import yaml
data = yaml.safe_load(f)
```

## Circular Imports

Extract shared types to a separate module rather than using lazy imports or `sys.path` manipulation:

```python
# Bad - lazy import to avoid circular
def get_thing():
    from pivot.other import Thing  # Circular!
    return Thing()

# Good - extract to types module
# pivot/types.py
class Thing: ...

# pivot/module_a.py
from pivot.types import Thing
```

## Mental Model Mistakes

### 1. Over-Engineering

Adding validation modes, config options, or abstractions for hypothetical future needs. If you can't articulate when each option would be used, you don't need options.

### 2. Defensive Over-Programming

Try/catch that returns a default on both paths; "safe" wrappers that swallow errors and hide bugs. Let errors surface.

### 3. Stage-Centric Thinking

**Wrong:** "Stage A triggers Stage B"
**Right:** "This artifact changed. What needs to happen because of that?"

The DAG emerges from artifact dependencies, not explicit stage wiring.

### 4. Type Safety Regression

Defaulting to `Any` when typing gets hard. Find the correct type or use TypeVar. `Any` is a last resort with documented justification.

## See Also

- [Code Style](style.md) - Coding conventions
- [Testing Guide](testing.md) - Writing tests
- [Architecture: Fingerprinting](../architecture/fingerprinting.md) - How fingerprinting works
