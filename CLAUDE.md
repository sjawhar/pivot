# Pivot - Project Rules

**Python Version:** 3.13+ | **Platform:** Unix only (Linux/macOS) | **Coverage Target:** 90%+

---

## Modifying CLAUDE.md

When updating CLAUDE.md files: **brevity and clarity over verbosity**. Reference existing code examples instead of writing full code blocks. Point to specific test files or functions. Strong, clear rules in few words beat lengthy explanations.

---

## Core Design

- Pivot eliminates DVC bottlenecks via per-stage lock files (32x faster), automatic code fingerprinting, and warm worker pools.
- Uses **ProcessPoolExecutor** for true parallel execution (not threads - GIL would serialize CPU work).
- TDD: write tests BEFORE implementation; 90%+ code coverage required.

## Stage Registration Patterns (Critical)

**Pivot supports three ways to define pipelines. All three must be considered when working on registry/discovery code:**

1. **`pivot.yaml`** - Configuration file with `stages:` section. Most common for teams.
   ```yaml
   stages:
     train:
       python: stages.train:run
       deps: [data/train.csv]
       outs: [models/model.pkl]
   ```

2. **`pipeline.py`** - Python script using `@stage` decorators or `Pipeline` class. Executed via `runpy.run_path()`.
   ```python
   # Using decorators
   from pivot.registry import stage
   @stage(deps=['x'], outs=['y'])
   def process(): ...

   # Or using Pipeline class
   from pivot import Pipeline
   pipeline = Pipeline()
   pipeline.add_stage(process, deps=['x'], outs=['y'])
   ```

3. **`@stage` decorators in importable modules** - Registered when modules are imported.

**Auto-discovery order:** `pivot.yaml` → `pivot.yml` → `pipeline.py` → decorator modules

When implementing features that reload or manipulate the registry (like reactive mode), all three patterns must be handled.

## Stage Function Requirements (Critical)

Stage functions must be **pure, serializable functions** for multiprocessing:

1. **Module-level definition** - Not lambdas, closures, or defined in `__main__`
2. **Picklable** - Function and all default arguments must serialize
3. **Pure** - No reliance on global mutable state (each process has its own copy)

```python
# Good - module-level function in importable module
@stage(deps=['data.csv'], outs=['output.csv'])
def process_data():
    import pandas as pd
    df = pd.read_csv('data.csv')
    df.to_csv('output.csv')

# Bad - lambda (not picklable)
process = stage(deps=['x'], outs=['y'])(lambda: ...)

# Bad - closure capturing local variable
def make_stage(threshold):
    @stage(deps=['x'], outs=['y'])
    def process():
        if value > threshold:  # Captures threshold - not picklable!
            ...
    return process

# Bad - defined in __main__ (no module path for pickle)
if __name__ == '__main__':
    @stage(deps=['x'], outs=['y'])
    def my_stage():  # Can't be pickled!
        ...
```

**Why?** Pivot uses `ProcessPoolExecutor` with `forkserver` context for true parallelism. Workers are separate processes that receive serialized (pickled) functions.

## StageDef Conventions (Typed Deps/Outs)

When using `StageDef` for typed dependencies and outputs:

1. **StageDef classes must be module-level** - Not defined inside functions (required for proper type hint resolution and pickling)
2. **Custom loaders must be module-level** - Same reason as above
3. **Loaders are frozen dataclasses** - Immutable for consistent fingerprinting
4. **Generic type parameter is for IDE/type-checker only** - The `T` in `CSV[pd.DataFrame]` enables autocomplete but isn't enforced at runtime

```python
from pivot import loaders
from pivot.stage_def import StageDef

# Good - module-level StageDef
class TrainParams(StageDef):
    class deps:
        data: loaders.CSV[pd.DataFrame] = "data/train.csv"
    class outs:
        model: loaders.Pickle = "models/model.pkl"
    learning_rate: float = 0.01

# Bad - inside function (type hints won't resolve)
def make_params():
    class BadParams(StageDef):  # Don't do this!
        class deps:
            data: loaders.CSV[pd.DataFrame] = "data.csv"
```

**Loader fingerprinting:** Loader code (the `load()`/`save()` methods) is fingerprinted. If you change loader behavior, stages using that loader will re-run.

**YAML overrides:** When `deps` or `outs` are specified in `pivot.yaml`, they completely replace the StageDef defaults (no merging).

## Code Quality Standards

- No code duplication; type hints everywhere; `ruff format` (line length 100); `ruff check` for linting.
- Concise one-line docstrings preferred; comments explain WHY not WHAT.
- Private functions use leading underscore; import modules not functions (Google style).
- No circular dependencies; higher-level modules depend on lower-level.

## Linting/Type Config (Critical)

- NEVER modify linting/type rules in `pyproject.toml` without explicit permission.
- **Zero tolerance for type checker warnings.** Both errors AND warnings from basedpyright must be resolved. Warnings indicate real type safety issues (missing annotations, unsafe operations, etc.) and are not acceptable.
- **No blanket pyright suppressions.** Don't use file-level `# pyright: reportFoo=false` to silence entire categories. Use targeted inline ignores on specific lines with explanations.
- **Always narrow type ignore comments.** Use specific error codes and explain why:
  ```python
  # Good - specific code with explanation
  return json.load(f)  # type: ignore[return-value] - json returns Any, user specifies T
  params._load_deps(root)  # pyright: ignore[reportPrivateUsage] - internal API

  # Bad - blanket ignore
  return json.load(f)  # type: ignore
  # pyright: reportPrivateUsage=false  # at file level
  ```
- **Prefer better type stubs over ignores.** Before adding a type ignore, check if the library has type stubs available (e.g., `pandas-stubs`, `types-PyYAML`). Install them if available to improve type coverage.

## Python 3.13+ Type Hints (Critical)

- Constructor syntax for empty collections: `list[int]()` not `: list[int] = []`
- Simplified Generator: `Generator[int]` not `Generator[int, None, None]`
- Use `Callable` instead of `Any` for callables; `Any` only as last resort (dynamic introspection, JSON parsing, testing).
- Use `TypeVar` for type-preserving decorators (preserves exact function signature).
- Use `Callable[..., Any]` for function params, not bare `Any`.
- Document why when using `Any`.

## TypedDict Usage (Critical)

**Why TypedDict over dataclasses/namedtuples:** TypedDict has zero runtime overhead (it's just a `dict`), native JSON serialization without conversion, and no boundary translation needed when parsing config files or API responses. Dataclasses require `asdict()` for JSON; namedtuples serialize as arrays, not objects. For structured data that flows to/from JSON, TypedDict avoids conversion overhead entirely.

**Never use `.get()` on TypedDicts.** TypedDicts have known keys at type-check time—use direct key access.
TypedDicts with `total=False` or `NotRequired` fields may have optional fields, in which case you should
use `if "key" in dict: ...` to check for the key and then use `dict["key"]` to access the value.

**Always use constructor syntax when creating TypedDicts**—not dict literals. This provides type validation at the call site and makes the code self-documenting. Applies everywhere: return statements, assignments, function arguments, and tests.

```python
class ChangeCheckResult(TypedDict):
    changed: bool
    reason: str

# Good - constructor syntax (type-checked)
return ChangeCheckResult(changed=True, reason="code changed")
result = ChangeCheckResult(changed=False, reason="")
some_function(ChangeCheckResult(changed=True, reason="test"))

# Bad - dict literal (no type validation, missing fields not caught)
return {"changed": True, "reason": "code changed"}
result = {"changed": False}  # Missing 'reason' not caught!
```

## Pydantic for Structured Data with Schemas

**Use Pydantic models for data that needs validation with clear error messages.** Configuration files, user-provided parameters, and API request/response bodies benefit from Pydantic's automatic validation, type coercion, and structured error messages.

```python
import pydantic

class RemoteTransferConfig(pydantic.BaseModel):
    """Remote transfer configuration."""
    jobs: Annotated[int, pydantic.Field(gt=0)] = 20
    retries: Annotated[int, pydantic.Field(ge=0)] = 10

# Pydantic validates on construction and gives clear errors
config = RemoteTransferConfig(jobs=-1)  # ValidationError: jobs must be > 0
```

**When to use Pydantic vs TypedDict:**

- **Pydantic:** Config files, user input, API boundaries—anywhere you need validation and helpful error messages
- **TypedDict:** Internal data structures, hot paths, data that's already validated upstream

**Exception:** Avoid Pydantic for hot paths where performance is critical (e.g., per-file hash lookups during pipeline execution). Use TypedDict or plain dicts there since Pydantic adds validation overhead.

## Import Style (Google Python Style Guide)

- Import modules, not functions: `from pivot import fingerprint` then `fingerprint.get_stage_fingerprint(...)`.
- No relative imports; no `sys.path` modifications.
- **This applies to all packages including third-party**—use `import pydantic` then `pydantic.BaseModel`, not `from pydantic import BaseModel`.
- No exceptions for stdlib—use `import pathlib` then `pathlib.Path`, not `from pathlib import Path`.
- Exception: type hints in `TYPE_CHECKING` blocks import classes/types directly, never modules. Use `from pathlib import Path`, not `import pathlib`. TYPE_CHECKING blocks are for type annotations only, so import the specific types you need. Never import the same thing both inside and outside TYPE_CHECKING.
- Exception: types from `pivot.types` may be imported directly: `from pivot.types import StageStatus, StageResult`.
- Exception: the `typing` module—always use `from typing import X`, never `import typing`. This is because `typing` exports are all type-related utilities meant to be used directly.

```python
# Good
from pivot import fingerprint
from pivot.types import StageStatus, StageResult
from typing import Any, TypeGuard  # typing is the exception - import directly
import pathlib
import pydantic

fp = fingerprint.get_stage_fingerprint(func)
path = pathlib.Path("/some/path")
status = StageStatus.READY

class MyParams(pydantic.BaseModel):
    learning_rate: float = 0.01

# Bad
from pivot.fingerprint import get_stage_fingerprint
from pathlib import Path
from pydantic import BaseModel
import typing  # Don't import typing as a module

fp = get_stage_fingerprint(func)  # Where is this from?
path = Path("/some/path")  # Where is Path from?
typing.get_type_hints(func)  # Don't use typing.X

class MyParams(BaseModel):  # Where is BaseModel from?
    learning_rate: float = 0.01
```

## No `__all__` Declarations

- Don't use `__all__` in modules—it's unnecessary maintenance overhead.
- Use underscore prefix (`_helper`) to indicate private functions.
- For package `__init__.py` re-exports, use explicit re-export syntax: `from module import X as X`.

## No Tiny Wrapper Functions

**Never create 1-2 line functions that just wrap a third-party library call—call the library directly.**

## Docstrings

**No module-level docstrings.** Don't add docstrings at the top of Python files unless explicitly asked.

**Simple functions (<20 lines) get one-line docstrings. Period.**

Skip Args/Returns/Examples if type hints make it obvious. Don't repeat what the function signature already says.

```python
# Bad - verbose, repeats type hints
def resolve_path(path: str) -> pathlib.Path:
    """Resolve path relative to project root (or use absolute).

    Args:
        path: File path (relative or absolute)

    Returns:
        Resolved absolute path
    """

# Good - concise, says what it does
def resolve_path(path: str) -> pathlib.Path:
    """Resolve relative path from project root; absolute paths unchanged."""

# Bad - explains obvious behavior
def get_project_root() -> pathlib.Path:
    """Get the project root directory.

    Returns the cached project root if available, otherwise finds it
    by calling find_project_root() and caches the result.

    Returns:
        Path: The project root directory path
    """

# Good - concise
def get_project_root() -> pathlib.Path:
    """Get project root (cached after first call)."""
```

## Comments - Code Clarity Over Comments (Critical)

**Prefer improving code clarity over leaving comments. Comments are a last resort.**

Before adding a comment, ask: "Can I make this code self-explanatory instead?" Better variable names, smaller functions, and clearer structure beat comments every time.

**When to add comments (rare):**

- Non-obvious WHY that can't be expressed in code
- Important timing/ordering constraints
- Known limitations or edge cases
- Complex algorithm steps that can't be simplified

**When NOT to add comments (common):**

- Obvious WHAT the code does (e.g., "Add node to graph" before `graph.add_node()`)
- Step-by-step descriptions (e.g., "Step 1:", "Step 2:", "Check if ready")
- Redundant with function/variable names (e.g., "Build output map" before `_build_outputs_map()`)
- Information already in docstrings
- Anything the code already expresses clearly

```python
# Good - explains non-obvious WHY
# Validate paths BEFORE normalizing (check ".." on original paths)
_validate_stage_registration(stages, name, deps, outs)

# Bad - states obvious WHAT
# Add all stages as nodes
for stage in stages:
    graph.add_node(stage)

# Good - clarifies intent
# Reverse chain to get correct order (module.sub.attr)
attr_path = ".".join(reversed(chain))

# Bad - redundant with code
# Parse source code to AST
tree = ast.parse(source)
```

**Rule of thumb:** If removing the comment makes the code unclear, try to write clearer code instead (helper functions, variable names). If that doesn't work, keep the comment.

**Write evergreen documentation:** Avoid time-relative language like "recently added", "new feature", "will be removed soon", or "as of version X". Future readers won't know when "recently" was. Instead, state facts that remain true regardless of when the documentation is read.

## Early Returns (Reduce Nesting)

- Use early `return`/`continue` to keep main logic at top indentation level.
- Avoid pyramid of doom; each guard clause should be simple, independent.

## Match Statements (Prefer Over if/elif Chains)

Use `match` statements instead of `if`/`elif` chains when dispatching on enum values, type discrimination, or multiple conditions on the same variable.

```python
# Good - match statement
match change["change_type"]:
    case ChangeType.ADDED:
        row_data.append(f"[green]{new_val}[/]")
    case ChangeType.REMOVED:
        row_data.append(f"[red]{old_val}[/]")
    case _ if old_val != new_val:
        row_data.append(f"[yellow]{old_val} -> {new_val}[/]")
    case _:
        row_data.append(str(new_val) if new_val is not None else "")

# Bad - if/elif chain for type dispatch
if change["change_type"] == ChangeType.ADDED:
    row_data.append(f"[green]{new_val}[/]")
elif change["change_type"] == ChangeType.REMOVED:
    row_data.append(f"[red]{old_val}[/]")
elif old_val != new_val:
    row_data.append(f"[yellow]{old_val} -> {new_val}[/]")
else:
    row_data.append(str(new_val) if new_val is not None else "")
```

**When to use match:** Enum dispatch, type discrimination, structured pattern matching, multiple conditions on one variable.

**When if/elif is fine:** Simple boolean conditions, early returns/guards, conditions on different variables.

## Private Functions

- Use `_prefix` for module-internal helpers; public = used by other modules.

## Error Handling

```python
# Good - specific exception
class FingerprintError(Exception): pass

# Bad - generic
raise Exception("failed")
```

**Let errors propagate.** The goal is not to prevent code from raising exceptions. Exceptions are valuable signals—they should propagate naturally and be caught at the appropriate boundary where they can be handled meaningfully.

**Principles:**

- Don't catch exceptions just to swallow them
- Catch exceptions where you can actually do something useful (retry, fallback, user message)
- Silent failures are worse than loud failures
- A crash with a stack trace is better than corrupted state

```python
# Good - let it propagate, catch at CLI boundary
def run_pipeline(stages: list[str]) -> dict[str, Result]:
    graph = build_dag(stages)  # May raise StageNotFoundError
    return execute(graph)      # May raise ExecutionError

# CLI catches and formats for user
try:
    results = run_pipeline(args.stages)
except StageNotFoundError as e:
    click.echo(f"Error: {e}", err=True)
    sys.exit(1)

# Bad - swallowing exceptions
def run_pipeline(stages: list[str]) -> dict[str, Result] | None:
    try:
        graph = build_dag(stages)
        return execute(graph)
    except Exception:
        return None  # Caller has no idea what went wrong
```

## Enums vs Literals

**Prefer enums for fixed sets of programmatic values.** Enums provide namespacing, iteration, IDE autocomplete, and catch typos at type-check time.

```python
# Good - enum for programmatic values
class CheckoutMode(enum.StrEnum):
    COPY = "copy"
    HARDLINK = "hardlink"
    SYMLINK = "symlink"

def checkout(mode: CheckoutMode) -> None:
    match mode:
        case CheckoutMode.COPY: ...
        case CheckoutMode.HARDLINK: ...
        case CheckoutMode.SYMLINK: ...

# Bad - literals scattered around
def checkout(mode: Literal["copy", "hardlink", "symlink"]) -> None:
    if mode == "copy": ...  # Typo "coppy" not caught until runtime
```

**When Literals are appropriate:**

- TypedDict discriminator fields (`type: Literal["log", "status"]`)
- JSON/YAML config values that map directly to strings
- Interop boundaries where you need raw strings

## Input Validation Over Defensive Programming

Prioritize validating inputs at boundaries rather than defensively handling every possible error deep in the code.

**Principles:**

- Validate inputs early, then trust them downstream
- Fail fast with clear errors rather than silently degrading
- Don't try to "fix" invalid inputs—reject them
- Avoid excessive try/catch that masks bugs

```python
# Good - validate input, then operate confidently
def export_stages(stages: list[str]) -> None:
    missing = [s for s in stages if s not in REGISTRY]
    if missing:
        raise ExportError(f"Stages not found: {missing}")

    for stage in stages:
        _export_stage(REGISTRY[stage])  # Known to exist

# Bad - defensive checks everywhere
def export_stages(stages: list[str]) -> None:
    for stage in stages:
        if stage in REGISTRY:  # Silent skip
            try:
                _export_stage(REGISTRY.get(stage, {}))
            except Exception:
                pass  # Silent failure
```

**When defensive code IS appropriate:**

- External I/O (network, filesystem, user input)
- Third-party library calls with unclear error modes
- Graceful degradation in non-critical paths

## Logging

```python
# Good
logger.info(f"Stage {name} completed in {duration:.2f}s")

# Bad
print(f"Stage {name} completed")
```

## CLI Output: Explicit Over Implicit

**Always show explicit messages instead of showing nothing.** When a user requests information and there's nothing to show, tell them explicitly rather than displaying nothing.

```python
# Good - explicit empty state
click.echo("Tracked Files")
if not tracked_status:
    click.echo("  No tracked files")
    return

# Bad - shows nothing for empty state
if tracked_status:
    click.echo("Tracked Files")
    for file in tracked_status:
        click.echo(f"  {file}")
```

**For JSON output:** Always include requested sections, even if empty. Use empty arrays/objects rather than omitting keys.

```python
# Good - always include requested keys
data = StatusOutput()
if show_stages:
    data["stages"] = pipeline_status  # Include even if empty list

# Bad - omits empty sections
if pipeline_status:  # Truthiness check excludes empty lists
    data["stages"] = pipeline_status
```

This applies to all user-facing output: CLI commands, status displays, list views, etc. Users should never wonder "did it work?" or "is something broken?" because the output was empty.

## CLI Shell Completion (Critical)

**All new CLI commands MUST include shell completion for stage/target arguments.** When adding a new CLI command:

1. **Stage arguments** (e.g., `run`, `status`, `explain`): Use `shell_complete=completion.complete_stages`
2. **Target arguments** (e.g., `push`, `pull`, `checkout`, `get`): Use `shell_complete=completion.complete_targets`

```python
from pivot.cli import completion

# For stage names
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
def my_command(stages: tuple[str, ...]) -> None:
    ...

# For targets (stages + file paths)
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
def my_command(targets: tuple[str, ...]) -> None:
    ...
```

This enables fast tab completion (~10ms) by parsing pivot.yaml directly without loading the full Pivot stack (~500ms).

## Development Commands

```bash
uv sync --active       # Install dependencies
pytest tests/ -n auto  # Run tests (parallel)
ruff format .          # Format
ruff check .           # Lint
basedpyright .         # Type check
```

## Pull Requests

- Use the PR template at `.github/pull_request_template.md`
- Include: Overview, Issue link, Approach/Alternatives, Testing, Checklist
- Highlight review focus areas (complex logic, edge cases, design decisions)
- Mark breaking changes and known limitations

## Before Returning to User (Critical)

- Must run all four: `ruff format .`, `ruff check .`, `basedpyright .`, `pytest tests/ -n auto`
- Never say "done" without running these first.

## Before Pushing to Git (Critical)

- ALWAYS run all quality checks before `jj git push`: `uv run ruff format . && uv run ruff check . && uv run basedpyright . && uv run pytest tests/ -n auto`
- CI will fail if checks don't pass locally.

## After Completing a Feature

- Update the **Development Roadmap** in `README.md` to reflect completed work
- Document any new **user-facing functionality** in the appropriate README section (CLI commands, API changes, new features)
- Keep the roadmap concise—use single-line summaries, not granular checklists

## Critical Discoveries

1. Test helpers must be module-level, not inline—`getclosurevars()` doesn't see module imports in inline closures.
2. Helper functions starting with single underscore (`_helper`) ARE tracked. Only dunder names (`__name__`, `__file__`, etc.) are filtered.
3. Use assertion messages instead of inline comments—messages appear in failure output.
4. **Circular import resolution:** When two modules have bidirectional dependencies, extract shared types/exceptions to a separate module (e.g., `exceptions.py` breaks `registry.py` ↔ `trie.py` cycle).
5. **AST manipulation validation:** After transforming AST nodes (e.g., removing docstrings), validate structural invariants. Function/class bodies must contain at least one statement—add `ast.Pass()` if empty.
6. **Path overlap detection requires Trie:** Simple string matching can't detect directory/file overlaps (`data/` vs `data/train.csv`). Use prefix-based data structure (pygtrie) for comprehensive validation.
7. **loky/cloudpickle can't pickle `mp.Queue()`:** Must use `mp.Manager().Queue()` for cross-process queues when using loky. Plain `mp.Queue()` fails with "Could not pickle the task to send it to the workers."
8. **Reusable executor pattern:** Use `loky.get_reusable_executor()` instead of creating new `ProcessPoolExecutor` instances—workers persist across calls, avoiding repeated import overhead.
9. **Queue readers must catch specific exceptions:** Only catch `queue.Empty` in queue polling loops, not broad `Exception`. Broad catches mask bugs and make debugging impossible.
10. **Cross-process tests need file-based state:** Shared mutable state (`execution_log = list[str]()`) silently fails in multiprocessing tests—each process gets its own copy. Use file-based logging (`open("log.txt", "a").write(...)`) for reliable cross-process communication in tests.
11. **Atomic file writes need `fd_closed` tracking:** When using `tempfile.mkstemp()` + rename pattern, track whether the fd was closed by a callback (e.g., `os.fdopen()`). Without this, the `finally` block may double-close or leak the fd.
12. **IncrementalOut must use COPY mode:** `IncrementalOut` restores from cache before execution so stages can modify in-place. Must use `LinkMode.COPY` (not hardlinks/symlinks) to avoid corrupting the cache when the stage writes to the file.
13. **Sentinel file locking for concurrent execution:** Use PID-based sentinel files (`stage.lock.running`) instead of `flock` for cross-platform support, visible debugging (can `ls` to see locks), and crash recovery via PID checking. Atomic acquisition via `os.open(path, O_CREAT | O_EXCL | O_WRONLY)`.
14. **Fingerprinting: `is_user_code()` check required for both import styles:** Direct references (`from utils import func`) and module attributes (`import utils; utils.func()`) both need `is_user_code()` check before hashing. The actual hash+recurse logic is centralized in `_add_callable_to_manifest()`.
15. **StateDB uses different path strategies for hash vs generation keys:** `_make_key_file_hash()` uses `path.resolve()` (follows symlinks) for physical file deduplication—multiple symlinks to the same file share one cached hash. `_make_key_output_generation()` uses `normpath(absolute())` (preserves symlinks) for logical path tracking—Pivot outputs become symlinks to cache after execution, and `resolve()` would follow those to cache paths that change per-run, breaking generation tracking.
16. **LMDB for all persistent caching:** Pivot uses LMDB (`.pivot/state.lmdb/`) with key prefixes (`hash:`, `gen:`, `dep:`, `remote:`) for all persistent state: file hash caching, generation counters, dependency tracking, and remote index. Prefer extending StateDB with new prefixes over adding new database technologies (e.g., SQLite, diskcache).
17. **ruamel.yaml for user-edited config, PyYAML for read-only:** Use `ruamel.yaml` (with `typ="rt"`) for config files that users edit directly—it preserves comments and formatting when modifying YAML. Use `PyYAML` for read-only YAML files like DVC pipelines where comment preservation doesn't matter. The `config/io.py` module demonstrates the pattern: `_load_config_preserving_structure()` preserves ruamel structure for comment-preserving edits, while `load_config_file()` converts to plain dict for general use.
18. **StageDef classes must be module-level:** When defining StageDef subclasses, always define them at module level, not inside test functions or other functions. `typing.get_type_hints()` requires the class's `__module__` to be importable for resolving forward references and generic type annotations. Classes defined inside functions have `__module__` set to the enclosing module but can't be found there, causing type hint resolution to fail with `Invalid loader annotation` errors.
19. **Stage failure leaves partial state (by design):** When a stage function raises an exception, outputs may be in a partial or incomplete state. This is acceptable behavior—Pivot prioritizes transparency over cleanup. Users can inspect partial outputs for debugging, and re-running the stage will produce fresh outputs. Do not add complex cleanup logic to hide failures; let them be visible.
