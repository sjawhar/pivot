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

### User Code Never Loads Through Mtime-Checked Bytecode

CPython reuses a cached `.pyc` while the source's size and whole-second mtime match the ones it recorded, so an edit that keeps a file's size, written in the second its previous version was written and cached, imports the previous code. While a discovery's import session lasts (from that discovery until the next one ends it), every file fingerprinting treats as user code (`fingerprint.is_library_path` is false for its resolved path; under the project root, outside the tooling directories discovery skips) imports through `discovery._ProjectSourceLoader`: in the process that discovered, and in every worker, whose pool installs the same path hook as its initializer (`discovery.install_user_code_loader`). A file `is_library_path` counts as a library's, pivot's own packages under an editable install of pivot included, loads as CPython loads it and its bytecode is left alone; what it counts is stated once, in `packages/pivot/tests/fingerprint/README.md` decision 17. A worker imports by name a stage's module that discovery does not send by value (one outside the root, or under it in a `src` layout) and any module a stage body imports. The loader caches checked-hash bytecode (PEP 552), checked against the source's content on every import by any process, after replacing other bytecode for the file (removing it under `sys.dont_write_bytecode`). It compiles from source when it cannot write the cache (a read-only `__pycache__`), under `sys.dont_write_bytecode`, and under `--check-hash-based-pycs never`, which workers do not inherit. A library package symlinked into the root loads as CPython loads it. A process pivot does not start, such as a subprocess a stage runs, imports with CPython's loader and can still read mtime-checked bytecode left in a read-only `__pycache__`. Before the first discovery, or after a session ends, imports get CPython's default loader: a test that rewrites a module and imports it outside a session, for example with `importlib.util.spec_from_file_location`, should remove `importlib.util.cache_from_source(path)` first.

Fingerprinting's own caches, of function hashes (`fp:`) and stage manifests (`sm:`), record each source file by the digest of its content, not by its mtime, size and inode, which an edit can keep to the nanosecond (two writes inside one kernel timestamp tick, or a tool that restores the mtime).

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

### Stage Functions Belong at Module Level

Workers receive stage functions by pickling (loky uses cloudpickle). A `@stage` defined inside a factory function does pickle and run, and the values it captures are fingerprinted, but configuration passed that way bypasses `StageParams`: it is not shown by `pivot params`, cannot be overridden in `params.yaml`, and the stage name no longer says which configuration it ran with. Define stages at module level and pass configuration as `StageParams`.

```python
# Good
class ProcessParams(StageParams):
    threshold: float = 0.5


@stage
def process(data: pd.DataFrame, params: ProcessParams) -> pd.DataFrame: ...


# Bad - configuration hidden in a closure
def make_stage(threshold):
    @stage
    def process(data: pd.DataFrame) -> pd.DataFrame:
        return data[data["value"] > threshold]

    return process
```

## Types

### Return-Type TypedDicts Should Be Module-Level

With postponed/string annotations (for example, `from __future__ import annotations`), Pivot resolves a stage's return annotation with `get_type_hints()`, which looks names up in the function's module namespace. A `TypedDict` defined inside a function is not there, and the pipeline fails to load with `name 'ProcessOutputs' is not defined`. Without postponed annotations Python 3.13 can preserve a function-local `TypedDict`, but module-level definitions work in both cases and are the project convention.

```python
# Good - module level
class ProcessOutputs(TypedDict):
    rows: pd.DataFrame
    summary: Annotated[dict, metric]


@stage
def process(data: pd.DataFrame) -> ProcessOutputs: ...


# Bad with postponed annotations - inline
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

### Incremental Outputs Use COPY Mode

An `Annotated[T, incremental]` output restores its previous cached artifact as a
writable copy before the stage receives the value bound by `pipeline.previous()`.
Copy mode prevents in-place changes from corrupting the cached version.

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
