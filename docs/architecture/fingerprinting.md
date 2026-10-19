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
| Imports in a function body | Yes | `import helpers` and `from helpers import scale` in a stage or helper body are resolved (relative ones against the function's package) and walked as a module-level import would be; a library module adds nothing |
| Class definitions | Yes | Full class AST hashed |
| Methods of user classes | Yes | Each function a user class's source defines (method, classmethod, staticmethod, property accessor, `functools.cached_property`), its own and inherited from user bases, is walked like a stage helper: the module values it reads are tracked or refused and the helpers it calls are walked. Generated methods (a dataclass's `__init__`) and library methods are not |
| StageParams subclasses | Yes | Type hint analysis + full class AST + Pydantic field defaults + `default_factory` functions |
| Custom loaders | Yes | Loader method ASTs hashed. A user loader's own `load()`, `save()` and `empty()`, and functions and classes in its config fields, are walked like stage helpers: module values they read are tracked or refused. Methods inherited from `pivot.loaders` are hashed only |
| Module-level immutable values | Yes (value) | Scalars, exact tuples and frozensets of scalars, and compiled regexes are recorded by value |
| Other module-level values | No — error | Use `StageParams` instead. Lists, dicts, sets, nested collections, enum members, and class instances are refused. This includes a method bound to an instance (`apply = Scaler(3).apply`, `normal = np.random.default_rng(SEED).normal`): the instance's state is not hashed. A method bound to a module, a class, a scalar or a `logging.Logger` is allowed |
| Closure variables | Yes (value) | `getclosurevars()` nonlocals checked like module-level values, keyed under the binding the closure was reached through, so two closures from one factory (`h2 = make(2)`, `h5 = make(5)`) keep their own values |
| Default arguments | Yes (value) | `__defaults__`/`__kwdefaults__` checked like module-level values; a literal default is covered by the AST hash |
| `functools.partial` | Yes | Bound args + underlying function; functions and classes in bound args walked like stage helpers |
| Library functions and classes | Yes (name) | A global, closure cell, default or partial holding `np.mean` records `ext:<binding>` → `numpy.mean`, so rebinding it to `np.median` reruns the stage; library code itself is not hashed |
| Secrets outside a stage's params | Yes (digest) | A `SecretStr`, `SecretBytes` or `Secret[T]` in a `StageParams` default on a helper, a loader config field or a partial argument hashes as its keyed `hmac:` digest, the one the lock file records (`parameters.secret_digest`), so changing it reruns the stage without its value in any manifest |
| Class attributes a stage reads | Yes (value) | `Cls.NAME` read directly on a user class that the function names (a global, cell or default holding the class), or reaches through a user module (`settings.Config.SCALE`). When `NAME` is written in a user class body, it follows the module-level rule: scalars, exact tuples and frozensets of scalars, and compiled regexes are recorded; mutable or nested values are refused; a plain function is walked; and a member of a user Enum is recorded by name and value. A pydantic or StageParams class read this way is covered by its schema hash, field defaults included. Attributes a library base or the language defines (an inherited stdlib dict, `NamedTuple._fields`, an Enum's `_member_names_`), and descriptors (property, classmethod, staticmethod), are left to the class AST hash |
| Other class-level and module reads | No | Not tracked: `cls.SCALE` inside a classmethod, a nested class (`Outer.Inner.X`), an attribute read through an instance (`Config().SCALE`), a `ClassVar` on a `StageParams` class, a module read through a local alias (`s = settings; s.SCALE`), a literal `getattr(settings, "SCALE")`, and a module held in a class attribute (`Holder.lib.SCALE`). Put such configuration in the stage's `StageParams` |

**Note:** The class AST and the method walk cover a class's code and the module values its methods read, but *runtime values* of class-level variables are tracked only in the `Cls.NAME` and `module.Cls.NAME` shapes above, and for Pydantic model field defaults. A value read as `module.NAME` follows the rule it follows as a global: a library function is an `ext:` entry, a `functools.partial` is walked, a logger is ignored.

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

### Imports by a Runtime Name

```python
def process():
    return importlib.import_module(name).run()  # Module chosen at run time
```

An import whose module is named at run time cannot be resolved statically:
`importlib.import_module` under any name and `__import__` are refused with
`StageDefinitionError`. A static import inside the function body
(`from helpers import process_data`) is tracked.

**Workaround:** Import the module statically.

### Module-Level Instances

```python
processor = DataProcessor()

def process():
    return processor.transform(data)  # State held by a module-level instance
```

A user class instance held at module level is refused: its attributes are state no code hash
records. An instance the stage creates (`DataProcessor().transform(data)`) is tracked through
the class's methods.

**Workaround:** Create the instance in the stage, with its settings from `StageParams`.

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

### Module-Level Values

A stage may read a module-level `bool`, `int`, `float`, `str`, `bytes` or `None`; its current value is recorded with `repr()`, so editing it re-runs the stage. A `tuple` or `frozenset` whose members are all such scalars, and a compiled regular expression, are recorded by value too: the tuple keeps member order, the frozenset sorts members by canonical JSON text (independent of `PYTHONHASHSEED`), and the regex records its full pattern and flags. Members are checked one level deep only. Reading any other module-level value (list, dict, set, a tuple or frozenset holding a tuple, list, dict or object, enum member, class instance, or a method bound to an instance such as `np.random.default_rng(SEED).normal`), directly or as `module.NAME`, is a `StageDefinitionError` that names the variable and points to `StageParams`. A default argument such as `cols=COLUMNS` or `factor=settings.FACTOR` on the stage or on a helper it calls follows the same rule, because a default is evaluated once when the module loads and its value never appears in the AST:

```python
THRESHOLD = 0.5            # allowed: tracked by value
COLUMNS = ("a", "b")       # allowed: tracked by value
STMT = re.compile(r"\s+")  # allowed: tracked by pattern and flags
PAIRS = (("a", 1),)        # error when a stage reads it
CONFIG = load_config()     # error when a stage reads it (a dict)
```

Configuration belongs in the stage's `StageParams`:

```python
class ProcessParams(StageParams):
    threshold: float = 0.5
    columns: list[str] = ["a", "b"]


@stage
def process(data: pd.DataFrame, params: ProcessParams) -> pd.DataFrame:
    return data.loc[data["value"] > params.threshold, params.columns]
```

For an explicitly unsafe migration path, `core.unsafe_fingerprinting=true` or
`PIVOT_UNSAFE_FINGERPRINTING=1` downgrades the error to a warning. A stage whose
fingerprint needed that downgrade is marked (`RegistryStageInfo.unsafe_fingerprint`),
and neither `pivot repro` nor `pivot commit` files its result in the run cache or
restores one from it.

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
| `self:` | The stage function itself | `self:train` |
| `func:` | Helper functions (user code callables) | `func:myproject.helpers.process_data` |
| `class:` | Class definitions | `class:myproject.params.TrainParams` |
| `const:` | Immutable values read as globals or as `module.NAME` | `const:myproject.settings.THRESHOLD` |
| `default:` | Default argument values that the AST does not record | `default:myproject.stages.scale.factor` |
| `wire:` | Call-site artifact wiring: parameter bindings, collection kind/order, input paths | `wire:left` → `raw-a`, `wire:items` → `{"kind":"list","identities":["a","b"]}`, `wire:file:path` → `old.txt` |
| `partial:` | `functools.partial` bound arguments | `partial:myproject.stage.bound.args` |
| `ext:` | A library function or class a stage reads, named by its definition (its code is not hashed) | `ext:myproject.stages.agg` → `numpy.mean` |
| `loader:` | Loader class methods and config | `loader:pivot.loaders.CSV:load` |
| `schema:` | Pydantic model JSON schema | `schema:myproject.params.TrainParams` |
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
$ pivot status --explain

Stage: demo/train
  Status: WILL RUN
  Reason: Code changed

  Code Changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4
```

## Manual Invalidation

If automatic detection fails, manually invalidate:

```bash
# Force re-run of specific stage
pivot repro demo/train --force
```

## Further Reading

- [Test Suite](https://github.com/sjawhar/pivot/tree/main/tests/fingerprint) - Comprehensive behavior tests
- [Architecture Overview](overview.md) - System design and other components
