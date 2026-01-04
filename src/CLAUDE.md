# Pivot Source Code Documentation

**Directory:** `/workspaces/treeverse/pivot/src/pivot/`
**Updated:** 2026-01-04

---

## Module Organization

This directory contains all Pivot source code organized by functionality.

### Core Modules (Week 1-2)

#### `fingerprint.py` - Code Change Detection

**Purpose:** Detect changes in user-defined Python functions automatically

**Key Functions:**

- `get_stage_fingerprint(func) -> dict[str, str]` - Returns manifest of all dependencies
- `hash_function_ast(func) -> str` - AST-based hash (normalized)
- `is_user_code(obj) -> bool` - Filter out stdlib/site-packages

**Algorithm:**

1. Hash function itself using AST
2. Use `inspect.getclosurevars()` to find all referenced names
3. For each reference:
   - If callable and user code → hash and recurse
   - If module → AST scan for `module.attr` usage
   - If simple constant → capture value
4. Return manifest dictionary

**Assumptions:**

- User code is in cwd or subdirectories (not site-packages)
- Functions use standard Python name lookup (no `eval`/`exec`)
- Module.attr patterns detectable via AST walk

**Testing:** `tests/test_fingerprint.py`

---

#### `ast_utils.py` - AST Parsing Utilities

**Purpose:** Extract information from Python AST for fingerprinting

**Key Functions:**

- `extract_module_attr_usage(func) -> list[tuple[str, str]]` - Find `module.attr` patterns
- `get_function_ast(func) -> ast.FunctionDef` - Parse function to AST
- `normalize_ast(node) -> ast.AST` - Remove irrelevant nodes (docstrings, etc.)

**Assumptions:**

- Functions have accessible source code (`inspect.getsource()` works)
- AST patterns are reliable across Python versions

**Testing:** `tests/test_ast_utils.py`

---

#### `registry.py` - Stage Registry

**Purpose:** Collect and manage pipeline stages

**Key Classes:**

- `StageRegistry` - Global singleton managing all stages
- `stage` - Decorator dataclass for marking functions as pipeline stages

**Key Functions:**

- `register(func, name, deps, outs, params_cls)` - Register a stage
- `get(name) -> dict` - Retrieve stage info
- `list_stages() -> list[str]` - Get all stage names

**Stage Decorator Pattern (Dataclass):**

The `stage` decorator is implemented as a dataclass that acts as a decorator:

```python
F = TypeVar("F", bound=Callable[..., Any])

@dataclass
class stage:
    """Decorator for marking functions as pipeline stages."""
    deps: list[str] = field(default_factory=list)
    outs: list[str] = field(default_factory=list)
    params_cls: type[BaseModel] | None = None

    def __call__(self, func: F) -> F:
        """Register function as a stage."""
        REGISTRY.register(
            func,
            name=func.__name__,
            deps=self.deps,      # Uses instance's fields
            outs=self.outs,
            params_cls=self.params_cls,
        )
        return func  # Returns original unmodified
```

**Type Preservation:**
The `F` TypeVar ensures the decorator returns the exact same type as the input function:

```python
@stage(deps=['data.csv'])
def train(lr: float) -> dict[str, float]:
    return {"loss": 0.5}

# Type checker knows: train is Callable[[float], dict[str, float]]
# NOT just: train is Callable[..., Any]
result = train(0.01)  # Type checker knows result is dict[str, float]
```

**How it works:**

1. `@stage(deps=['data.csv'])` creates a `stage` instance with `deps=['data.csv']`
2. The instance's `__call__()` method receives the function
3. It registers the function with the deps/outs from the instance
4. Returns the original function unmodified

**Example:**

```python
@stage(deps=['data.csv'], outs=['model.pkl'])
def train():
    return 42

# Equivalent to:
decorator = stage(deps=['data.csv'], outs=['model.pkl'])  # Create instance
train = decorator(train)  # Call instance with function
```

**Benefits:**

- Clean syntax: `@stage(deps=[...])`
- Default values via dataclass fields
- Type safety built-in
- No manual `__init__` needed

**Stage Info Structure:**

```python
{
    'func': callable,                # The Python function
    'name': str,                     # Stage name (defaults to func.__name__)
    'deps': list[str],              # Input file/stage dependencies
    'outs': list[str],              # Output files
    'params_cls': type[BaseModel],  # Pydantic model (optional)
    'signature': Signature,          # Function signature (from inspect)
    'fingerprint': dict[str, str],  # Code dependency manifest (from fingerprinting)
}
```

**Assumptions:**

- Stages decorated before pipeline execution
- Stage names are unique
- Decorator doesn't modify function behavior
- Fingerprinting happens on registration (captures code state at that moment)

**Testing:** `tests/test_registry.py`

---

### Graph and Scheduling (Week 2, 5)

#### `dag.py` - Dependency Graph

**Purpose:** Build and analyze stage dependency graph

**Key Functions:**

- `build_dag(stages: dict) -> nx.DiGraph` - Construct DAG from stage definitions
- `topological_sort(dag, targets) -> list[str]` - Execution order
- `check_acyclic(dag)` - Detect cycles, raise exception
- `find_output_producer(stages, file_path) -> str | None` - Which stage creates a file

**Graph Structure:**

```python
# Nodes: stage names
# Edges: (downstream, upstream) - downstream depends on upstream
# Node data: full stage_info dict
```

**Assumptions:**

- File-based dependencies: one file = one producer
- Stage dependencies expressed as `stage:<name>` in deps list
- Graph fits in memory (reasonable for <10,000 stages)

**Testing:** `tests/test_dag.py`

---

#### `scheduler.py` - Parallel Execution Scheduler

**Purpose:** Execute stages in parallel respecting dependencies

**Key Classes:**

- `StageInfo` - Track execution state (adapted from DVC pattern)
- `StageStatus` - Enum: PENDING, RUNNING, COMPLETED, SKIPPED, FAILED

**Key Functions:**

- `execute_parallel(stages, workers, force, executor_type) -> dict[str, ExecutionResult]`
- `get_ready_stages(stage_info_map) -> list[str]` - Stages with no unfinished deps
- `handle_completion(stage_name, future)` - Update dependent stages

**Algorithm (DVC Pattern):**

1. Initialize StageInfo for each stage
2. Submit ready stages to executor
3. Wait for ANY completion (FIRST_COMPLETED)
4. Update dependent stages' `upstream_unfinished` sets
5. Submit newly-ready stages
6. Repeat until all done

**Assumptions:**

- Executor interface matches concurrent.futures.Executor
- Stage execution is idempotent (can retry on failure if implemented)
- Errors cascade downstream (dependent stages skip)

**Testing:** `tests/test_scheduler.py`

---

### Execution (Week 4-5)

#### `executor.py` - Sequential Execution

**Purpose:** Execute stages one at a time (validation baseline)

**Key Functions:**

- `execute_stage(stage_name, force) -> ExecutionResult` - Run single stage
- `execute_pipeline(stages, force) -> dict[str, ExecutionResult]` - Run in order

**Assumptions:**

- Used for testing and debugging (not production default)
- Simpler error messages than parallel execution

**Testing:** `tests/test_executor.py`

---

#### `executors.py` - Executor Selection

**Purpose:** Choose and configure execution backend

**Key Classes:**

- `ExecutorType` - Enum: WARM_WORKER, INTERPRETER, PROCESS
- `WarmWorkerPoolExecutor(ProcessPoolExecutor)` - With preloaded imports

**Key Functions:**

- `get_executor(type, max_workers, preload_modules) -> Executor`
- `initialize_warm_worker(preload_modules)` - Worker initializer

**Executor Characteristics:**

| Type        | Use Case                 | Pros                         | Cons                             |
| ----------- | ------------------------ | ---------------------------- | -------------------------------- |
| WARM_WORKER | Import-heavy (default)   | Preloaded imports, proven    | Higher memory                    |
| INTERPRETER | CPU-bound (experimental) | Lower memory, faster startup | No shared imports (Python 3.14+) |
| PROCESS     | Fallback                 | Always works                 | Cold starts                      |

**Assumptions:**

- `forkserver` context available (Linux/Mac)
- Preloaded modules picklable (numpy, pandas are)
- InterpreterPoolExecutor available on Python 3.14+

**Testing:** `tests/test_executors.py`

---

### Storage (Week 2-3)

#### `lock.py` - Per-Stage Lock Files

**Purpose:** Track stage execution state and detect changes

**Key Classes:**

- `StageLock` - Read/write lock files for a stage

**Key Functions:**

- `read() -> dict | None` - Load lock file
- `write(data: dict)` - Atomic write
- `is_changed(fingerprint, params, dep_hashes) -> tuple[bool, str]` - Detect changes

**Lock File Format:**

```yaml
schema: "1.0"
run_id: "abc123def"
combined_hash: "a1b2c3d4"

code_manifest:
  self:train: "fcbf9f57"
  func:helper_a: "5995c853"
  const:LEARNING_RATE: "0.01"

params:
  learning_rate: 0.01
  epochs: 200

deps:
  - path: data/train.csv
    hash: "xxh64:abc123"

outs:
  - path: model.pkl
    hash: "xxh64:def456"
```

**Change Detection Logic:**

1. Code manifest changed? → Re-run
2. Params changed? → Re-run
3. Input file hash changed? → Re-run
4. Output file missing? → Re-run
5. Otherwise → Skip

**Assumptions:**

- Atomic file replacement on POSIX (`.tmp` then `rename`)
- YAML safe_load/safe_dump sufficient (not round-trip)
- Lock files < 10KB (reasonable for fingerprint manifest)

**Testing:** `tests/test_lock.py`

---

#### `cache.py` - Content-Addressed Hashing

**Purpose:** Fast file and directory hashing

**Key Functions:**

- `hash_file(path: Path) -> str` - xxhash64 of file contents
- `hash_directory(path: Path) -> str` - Recursive directory hash

**Hash Format:** `"xxh64:<hex_digest>"`

**Assumptions:**

- xxhash sufficient for collision resistance (64-bit)
- Files fit in memory for streaming hash (8KB chunks)
- Directory hashing includes file names and contents

**Testing:** `tests/test_cache.py`

---

### Parameters (Week 3)

#### `params.py` - Pydantic Parameter System

**Purpose:** Auto-generate and validate stage parameters

**Key Functions:**

- `create_params_model(func) -> type[BaseModel]` - Generate from signature
- `load_params(params_model, yaml_file) -> BaseModel` - Load and validate

**Algorithm:**

1. Inspect function signature
2. Extract type hints and defaults
3. Generate Pydantic model dynamically
4. Load YAML and validate

**Example:**

```python
def train(lr: float = 0.01, epochs: int = 100):
    ...

# Auto-generates:
class TrainParams(BaseModel):
    lr: float = 0.01
    epochs: int = 100

# Load from params.yaml:
params = load_params(TrainParams, Path("params.yaml"))
```

**Assumptions:**

- Functions have type hints for parameters
- YAML file structure matches function signature
- Pydantic validation rules sufficient

**Testing:** `tests/test_params.py`

---

### Observability (Week 4)

#### `explain.py` - Explain Mode

**Purpose:** Show WHY a stage would run

**Key Functions:**

- `explain_stage(stage_name) -> dict` - Detailed explanation
- `print_explain(stages)` - Human-readable output

**Output Format:**

```
Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4
      File: src/utils.py:15
```

**Assumptions:**

- Lock file exists from previous run (or report "No previous run")
- Source file locations findable for functions

**Testing:** `tests/test_explain.py`

---

### Export (Week 2)

#### `export.py` - DVC YAML Export

**Purpose:** Generate DVC-compatible pipeline definition

**Key Functions:**

- `export_to_dvc_yaml(output_path) -> None` - Generate dvc.yaml
- `generate_command(stage_info) -> str` - Create `python -c "..."` command
- `validate_export(dvc_yaml_path) -> bool` - Run DVC and compare

**Generated Command Format:**

```python
python -c "import sys; sys.path.insert(0, '.'); from pipeline import train; import yaml; params = yaml.safe_load(open('params.yaml'))['train']; train(**params)"
```

**Assumptions:**

- Pipeline module importable from current directory
- params.yaml exists if stage has parameters
- DVC installed for validation

**Testing:** `tests/test_export.py`, `tests/integration/test_dvc_compatibility.py`

---

### CLI (Week 6)

#### `cli.py` - Command Line Interface

**Purpose:** User-facing commands

**Commands:**

- `pivot run [stages] [--workers N] [--force] [--explain] [--executor TYPE]`
- `pivot status` - Show which stages need re-run
- `pivot dag [--save FILE]` - Visualize pipeline
- `pivot export [--output FILE] [--validate]` - Generate dvc.yaml

**Assumptions:**

- Click for CLI framework
- Rich for colored output (optional)

**Testing:** `tests/test_cli.py`

---

## Module Dependencies

```
fingerprint.py  →  ast_utils.py
    ↓
registry.py  →  fingerprint.py, params.py
    ↓
dag.py  →  registry.py
    ↓
lock.py  →  cache.py
    ↓
executor.py  →  registry.py, lock.py, cache.py
    ↓
executors.py  →  (external: multiprocessing, concurrent.futures)
    ↓
scheduler.py  →  dag.py, executor.py, executors.py
    ↓
explain.py  →  lock.py, fingerprint.py
    ↓
export.py  →  registry.py
    ↓
cli.py  →  executor.py, scheduler.py, explain.py, export.py
```

**Rule:** No circular dependencies. Higher-level modules depend on lower-level.

---

## Code Standards

### Linting and Type Checking Configuration

**CRITICAL RULE:** Do NOT modify linting or type checking rules in `pyproject.toml` without explicit user permission.

**Prohibited actions:**

- Adding `# type: ignore` suppressions without justification
- Disabling basedpyright rules (e.g., `reportXXX = false`)
- Changing ruff configuration
- Lowering coverage thresholds

**When encountering type/lint errors:** Fix the code, don't silence the checker.

**Exception:** Well-justified suppressions with comments explaining why (e.g., `# type: ignore[import-not-found]  # Test-only module`).

---

### Import Style (Google Python Style Guide)

**Rule:** Import modules, not individual functions/classes. Use `from` imports only for:

1. Standard library modules
2. Type hints (in `if TYPE_CHECKING:` blocks)

**Import Requirements:**

- **No relative imports** - Use absolute imports: `from pivot import module`, not `from . import module`
- **No sys.path modifications** - Do not modify `sys.path` to fix import issues
- Exceptions: Internal package imports use relative imports (`from . import X`) to avoid cycles

**Good Examples:**

```python
# Import modules (PREFERRED)
import ast
import inspect
import logging
from pathlib import Path

from pivot import fingerprint
from pivot import registry

# Use qualified names
def analyze_function(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    fp = fingerprint.get_stage_fingerprint(func)
    return registry.REGISTRY.get(func.__name__)
```

**Bad Examples:**

```python
# Don't import functions/classes directly (AVOID)
from inspect import getsource, getclosurevars
from ast import parse, FunctionDef
from pivot.fingerprint import get_stage_fingerprint
from pivot.registry import REGISTRY, stage

# This makes it unclear where things come from
def analyze_function(func):
    source = getsource(func)  # Where is getsource from?
    tree = parse(source)       # Where is parse from?
    fp = get_stage_fingerprint(func)  # From pivot?
    return REGISTRY.get(func.__name__)  # What module is REGISTRY?
```

**Exceptions (Allowed):**

1. **Type hints only:**

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pivot.types import StageInfo
```

2. **Standard library when unambiguous:**

```python
from pathlib import Path  # OK - Path is unambiguous
from dataclasses import dataclass  # OK - decorator, clear usage
```

**Why This Style?**

1. **Clarity:** Always clear where a function/class comes from
2. **Namespace management:** Prevents naming conflicts
3. **Readability:** Easier to understand dependencies
4. **Refactoring:** Easier to move code between modules

**Reference:** https://google.github.io/styleguide/pyguide.html#22-imports

---

### Docstrings - Concise and Readable

**Rule:** Keep docstrings short and focused. One-line docstrings are preferred for simple functions.

**Good Examples:**

```python
# Good - concise one-liners
def hash_function_ast(func: Any) -> str:
    """Hash function AST (ignores whitespace, comments, docstrings)."""
    ...

def extract_module_attr_usage(func: Any) -> list[tuple[str, str]]:
    """Extract module.attr patterns (e.g., 'np.array') from function AST."""
    ...

def is_user_code(obj: Any) -> bool:
    """Check if object is user code (not stdlib/site-packages/builtins)."""
    ...

# Good - multi-line for complex functions (only when needed)
def get_stage_fingerprint(func: Any, visited: set[int] | None = None) -> dict[str, str]:
    """Generate fingerprint manifest capturing all code dependencies.

    Returns dict with keys:
    - 'self:<name>': Function itself
    - 'func:<name>': Referenced helper functions (transitive)
    - 'mod:<module>.<attr>': Module attributes ("callable" or value)
    - 'const:<name>': Global constants
    """
    ...
```

**Bad Examples:**

```python
# Bad - unnecessarily verbose for simple function
def hash_function_ast(func: Any) -> str:
    """Hash function using AST (normalized, ignores whitespace/comments).

    Produces stable hash that ignores:
    - Whitespace differences
    - Comments
    - Docstrings

    Variable names, logic, and structure ARE significant.

    Args:
        func: Function to hash

    Returns:
        Hex string hash of function AST (16 chars)

    Example:
        >>> def f(): return 42
        >>> h1 = hash_function_ast(f)
        >>> def f(): return 42  # Different whitespace
        >>> h2 = hash_function_ast(f)
        >>> h1 == h2
        True
    """
    ...  # Function is only 10 lines but docstring is 25!

# Bad - redundant documentation
def extract_module_attr_usage(func: Any) -> list[tuple[str, str]]:
    """Extract module.attr patterns from function.

    Scans function AST for patterns like:
    - numpy.array()
    - pd.DataFrame()
    - module.function()

    Args:
        func: Function to analyze

    Returns:
        List of (module_name, attr_name) tuples
    """
    ...  # Type hints already say this!
```

**Guidelines:**

- One-line docstrings for functions < 20 lines
- Avoid redundant Args/Returns if type hints are clear
- Skip Examples unless behavior is non-obvious
- Focus on "why" not "what" (code shows "what")

---

### Code Comments - WHY Not WHAT

**Rule:** Comments explain WHY the code does something non-obvious, not WHAT the code does. Code should be self-documenting through clear naming and structure.

**Good Examples:**

```python
# Good - explains WHY (non-obvious decision)
def _normalize_ast(node: ast.AST) -> ast.AST:
    """Remove docstrings and normalize function names for stable hashing."""
    # Normalize names to "func" so identical logic produces same hash
    if isinstance(node, ast.FunctionDef):
        node.name = "func"
    return node

# Good - explains WHY (surprising behavior)
def is_user_code(obj: Any) -> bool:
    """Check if object is user code (not stdlib/site-packages/builtins)."""
    # Skip underscore names to filter __name__, __file__, etc.
    if name.startswith("_"):
        return False
    ...

# Good - explains WHY (performance/correctness trade-off)
def hash_function_ast(func: Any) -> str:
    """Hash function AST (ignores whitespace, comments, docstrings)."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        # Builtins don't have source; use code object as fallback
        if hasattr(func, "__code__"):
            return hashlib.sha256(func.__code__.co_code).hexdigest()[:16]
        return hashlib.sha256(str(id(func)).encode()).hexdigest()[:16]
    ...
```

**Bad Examples:**

```python
# Bad - states the obvious (WHAT)
def hash_function_ast(func: Any) -> str:
    """Hash function AST."""
    try:
        # Get source code
        source = inspect.getsource(func)
    except (OSError, TypeError):
        # Can't get source
        # Fall back to hashing code object
        if hasattr(func, "__code__"):
            return hashlib.sha256(func.__code__.co_code).hexdigest()[:16]
        # No code available, use identity
        return hashlib.sha256(str(id(func)).encode()).hexdigest()[:16]

    # Parse to AST
    tree = ast.parse(source)

    # Normalize AST
    tree = _normalize_ast(tree)

    # Convert AST to string
    ast_str = ast.dump(tree)

    # Hash the AST string
    return hashlib.sha256(ast_str.encode()).hexdigest()[:16]

# Bad - redundant step-by-step narration
def extract_module_attr_usage(func: Any) -> list[tuple[str, str]]:
    """Extract module.attr patterns."""
    try:
        # Get source
        source = inspect.getsource(func)
    except (OSError, TypeError):
        # Can't get source
        return []

    # Parse to AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse
        return []

    # Walk AST looking for module.attr patterns
    attrs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        # Look for ast.Attribute nodes (module.attr)
        if isinstance(node, ast.Attribute):
            # Check if the value is a Name node (simple identifier)
            if isinstance(node.value, ast.Name):
                # Get module and attr names
                module_name = node.value.id
                attr_name = node.attr
                # Add to list
                attrs.append((module_name, attr_name))

    # Remove duplicates while preserving order
    seen = set()
    unique_attrs = []
    for attr in attrs:
        if attr not in seen:
            seen.add(attr)
            unique_attrs.append(attr)

    return unique_attrs
```

**When to Comment:**

- ✅ Explaining a non-obvious algorithm or approach
- ✅ Documenting assumptions or constraints
- ✅ Clarifying surprising behavior or edge cases
- ✅ Justifying a performance/correctness trade-off
- ✅ Warning about gotchas or common mistakes
- ❌ Restating what the code obviously does
- ❌ Narrating the code line-by-line
- ❌ Translating code to English
- ❌ Explaining standard library calls

**Self-Documenting Code:**

```python
# Good - clear without comments
def extract_module_attr_usage(func: Any) -> list[tuple[str, str]]:
    """Extract module.attr patterns (e.g., 'np.array') from function AST."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    attrs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                attrs.append((node.value.id, node.attr))

    seen = set()
    unique_attrs = []
    for attr in attrs:
        if attr not in seen:
            seen.add(attr)
            unique_attrs.append(attr)

    return unique_attrs
```

---

### Early Returns - Reduce Nesting

**Rule:** Use early returns and continues to reduce nesting depth. Exit guard clauses early instead of wrapping the main logic in nested conditions.

**Good Examples:**

```python
# Good - early returns reduce nesting
def process_item(item: dict[str, Any]) -> str | None:
    """Process item and return result."""
    if not item:
        return None

    if "name" not in item:
        return None

    if not item["name"].strip():
        return None

    # Main logic at top level
    result = item["name"].upper()
    return result


# Good - early continue in loops
def process_items(items: list[dict[str, Any]]) -> list[str]:
    """Process all valid items."""
    results = []

    for item in items:
        if not item:
            continue

        if "name" not in item:
            continue

        # Main logic at top level
        results.append(item["name"].upper())

    return results


# Good - guard clauses
def is_user_code(obj: Any) -> bool:
    """Check if object is user code."""
    if obj is None:
        return False

    module = _get_module(obj)
    if module is None:
        return False

    module_name = getattr(module, "__name__", "")
    if module_name in sys.builtin_module_names:
        return False

    if not hasattr(module, "__file__") or module.__file__ is None:
        return True

    # Main logic at top level
    module_file = Path(module.__file__).resolve()
    if _is_stdlib_path(module_file):
        return False

    return not any(path in module_file.parts for path in _SITE_PACKAGE_PATHS)
```

**Bad Examples:**

```python
# Bad - nested conditions create pyramid of doom
def process_item(item: dict[str, Any]) -> str | None:
    """Process item and return result."""
    if item:
        if "name" in item:
            if item["name"].strip():
                # Main logic buried 3 levels deep
                result = item["name"].upper()
                return result
            else:
                return None
        else:
            return None
    else:
        return None


# Bad - nested conditions in loop
def process_items(items: list[dict[str, Any]]) -> list[str]:
    """Process all valid items."""
    results = []

    for item in items:
        if item:
            if "name" in item:
                # Main logic nested 2 levels deep
                results.append(item["name"].upper())

    return results


# Bad - deeply nested logic
def is_user_code(obj: Any) -> bool:
    """Check if object is user code."""
    if obj is not None:
        module = _get_module(obj)
        if module is not None:
            module_name = getattr(module, "__name__", "")
            if module_name not in sys.builtin_module_names:
                if hasattr(module, "__file__") and module.__file__ is not None:
                    # Main logic buried 4 levels deep
                    module_file = Path(module.__file__).resolve()
                    if not _is_stdlib_path(module_file):
                        return not any(path in module_file.parts for path in _SITE_PACKAGE_PATHS)
                    return False
                else:
                    return True
            return False
        return False
    return False
```

**Benefits:**

- ✅ Main logic stays at top indentation level (easier to read)
- ✅ Guard clauses clearly show preconditions
- ✅ Reduces cognitive load (fewer nested contexts to track)
- ✅ Easier to add new conditions without increasing nesting
- ✅ Clear separation between validation and logic

**Guidelines:**

- Use early `return` for validation/error checks in functions
- Use early `continue` for validation/filtering in loops
- Keep main logic at the lowest nesting level possible
- Each guard clause should be a simple, independent check
- Avoid nesting beyond 2-3 levels

---

### Private Functions - Leading Underscore

**Rule:** Functions used only within their module should start with `_` to indicate they're internal.

**Good Examples:**

```python
# Public API - used by other modules
def get_stage_fingerprint(func: Any) -> dict[str, str]:
    """Generate fingerprint manifest capturing all code dependencies."""
    manifest = {}
    manifest["self"] = hash_function_ast(func)
    tree = _normalize_ast(ast.parse(source))  # Call private helper
    return manifest

# Private helper - only used within this module
def _normalize_ast(node: ast.AST) -> ast.AST:
    """Remove docstrings and normalize function names for stable hashing."""
    if isinstance(node, ast.FunctionDef):
        node.name = "func"
    return node
```

**Bad Examples:**

```python
# Bad - helper is public but only used internally
def normalize_ast(node: ast.AST) -> ast.AST:  # Should be _normalize_ast
    """Remove docstrings..."""
    ...

# Bad - public function that should be private
def extract_attrs_from_tree(tree: ast.AST) -> list:  # Only used in this module
    """Extract attributes..."""
    ...
```

**Guidelines:**

- Use `_prefix` for module-internal helpers
- Public functions: exported in `__all__` or used by other modules
- Private functions: only called within the same module
- Tests can still import private functions (that's OK)

---

### Type Hints - Prefer Specific Types Over Any

**Rule:** Use specific types whenever possible. `Any` should be reserved for truly polymorphic cases where the type can vary widely.

#### Type Preservation with TypeVar

Use `TypeVar` for type-preserving decorators and wrappers:

```python
from typing import TypeVar
from collections.abc import Callable

F = TypeVar("F", bound=Callable[..., Any])

@dataclass
class stage:
    """Decorator that preserves function type."""
    deps: list[str] = field(default_factory=list)
    outs: list[str] = field(default_factory=list)

    def __call__(self, func: F) -> F:  # Returns SAME type as input
        REGISTRY.register(func, deps=self.deps, outs=self.outs)
        return func

# Type checker knows exact type:
@stage(deps=['data.csv'])
def train(lr: float) -> dict[str, float]:
    return {"loss": 0.5}

# train is Callable[[float], dict[str, float]], NOT Callable[..., Any]
```

**Why TypeVar?** Preserves the exact function signature through the decorator. Without it, type checkers lose information about the wrapped function's parameters and return type.

#### Narrowing Function Parameters

Prefer `Callable[..., Any]` over bare `Any` for function parameters:

```python
# Good - specific type for function parameters
def get_stage_fingerprint(func: Callable[..., Any]) -> dict[str, str]:
    """Generate fingerprint for a callable function."""
    ...

def hash_function_ast(func: Callable[..., Any]) -> str:
    """Hash function AST."""
    ...

# Acceptable - truly polymorphic parameter
def is_user_code(obj: Any) -> bool:
    """Check if ANY object is user code (module, function, class, etc.)."""
    if isinstance(obj, ModuleType):
        ...
    elif callable(obj):
        ...
    # Handles many different types - Any is justified here
```

**Bad Examples:**

```python
# Bad - func parameter should be Callable[..., Any]
def get_stage_fingerprint(func: Any) -> dict[str, str]:
    """Generate fingerprint."""
    ...  # We know it's a function!

# Bad - decorator should preserve type
def stage_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register stage."""
    return func  # Lost the specific function type!
```

#### Guidelines

- **Use `Callable[..., Any]`** for function parameters (not `Any`)
- **Use `TypeVar`** for type-preserving decorators and wrappers
- **Use `Any`** only when the type truly varies widely (e.g., `obj: Any` that could be module, function, class, value)
- **Document why** when using `Any` - explain what types are expected
- **Import from `collections.abc`** for runtime type hints: `from collections.abc import Callable`

**When `Any` is Justified:**

1. Truly polymorphic functions that handle many unrelated types
2. Dynamic introspection where type depends on runtime behavior
3. Interfacing with untyped third-party code
4. Type is genuinely unknowable at static analysis time

#### Standard Type Annotations

```python
# Good - complete type hints
def get_stage_fingerprint(func: Callable[..., Any], visited: set[int] | None = None) -> dict[str, str]:
    ...

# Bad - missing types
def get_stage_fingerprint(func, visited=None):
    ...
```

---

### Docstrings (Google Style)

```python
def hash_file(path: Path) -> str:
    """Hash file contents using xxhash64.

    Args:
        path: Path to file to hash

    Returns:
        Hash string in format "xxh64:<hex>"

    Raises:
        FileNotFoundError: If file doesn't exist
    """
```

### Error Handling

```python
# Good: Specific exceptions
class FingerprintError(Exception):
    """Raised when fingerprinting fails."""
    pass

# Bad: Generic exceptions
raise Exception("Fingerprinting failed")
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Good: Structured logging
logger.info(f"Stage {name} completed in {duration:.2f}s")

# Bad: Print statements
print(f"Stage {name} completed")
```

---

## Next Steps

1. **Week 1:** Implement `fingerprint.py`, `ast_utils.py`, `registry.py`
2. **Validation:** Run tests, get user approval before Week 2
3. **Documentation:** Update this file if implementation diverges from plan

**See:** `tests/CLAUDE.md` for testing strategy details.
