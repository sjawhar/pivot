# Code Style

Coding conventions and patterns for Pivot development.

## General Rules

- **Type hints everywhere** - All functions must have type hints
- **100 character line limit** - Enforced by ruff
- **One-line docstrings** - Skip Args/Returns if type hints make it obvious
- **Comments explain WHY** - Not WHAT the code does

## Import Style

Import modules, not functions:

```python
# Good
import pathlib
import pandas
from pivot import fingerprint

path = pathlib.Path("/some/path")
df = pandas.read_csv("data.csv")
fp = fingerprint.get_stage_fingerprint(func)

# Bad
from pathlib import Path
from pandas import read_csv
from pivot.fingerprint import get_stage_fingerprint
```

**No lazy imports** - All imports at module level. This ensures fingerprinting captures dependencies and makes imports explicit.

**Exceptions:**

- `TYPE_CHECKING` blocks: Import types directly
- `pivot.types`: Import directly (`from pivot.types import StageStatus`)
- `typing` module: Always direct (`from typing import Any`)
- CLI modules: Lazy imports acceptable in `pivot.cli` to reduce startup time

## Private Functions

Use underscore prefix for module-internal helpers:

```python
def public_function():
    """Public API."""
    return _internal_helper()

def _internal_helper():
    """Not part of public API."""
    pass
```

## TypedDict Usage

Zero runtime overhead, native JSON serialization. Use over dataclasses or namedtuples.

Always use constructor syntax:

```python
class Result(TypedDict):
    status: str
    value: int

# Good
return Result(status="ok", value=42)

# Bad - no type validation
return {"status": "ok", "value": 42}
```

Never use `.get()` - direct access only. For optional fields:

```python
if "key" in d:
    value = d["key"]
```

## Early Returns

Use early returns to reduce nesting:

```python
# Good
def process(data: Data | None) -> Result:
    if data is None:
        return Result(status="error", value=0)
    # Main logic at top level
    return Result(status="ok", value=data.compute())

# Bad
def process(data: Data | None) -> Result:
    if data is not None:
        # Nested logic
        return Result(status="ok", value=data.compute())
    else:
        return Result(status="error", value=0)
```

## Match Statements

Prefer over if/elif for enum dispatch and type discrimination:

```python
match status:
    case StageStatus.SUCCESS:
        handle_success()
    case StageStatus.FAILED:
        handle_failure()
    case StageStatus.SKIPPED:
        handle_skip()
```

## Docstrings

No module-level docstrings. Simple functions get one-line docstrings:

```python
# Good
def resolve_path(path: str) -> pathlib.Path:
    """Resolve relative path from project root; absolute paths unchanged."""

# Bad - repeats type hints
def resolve_path(path: str) -> pathlib.Path:
    """Resolve path relative to project root.

    Args:
        path: File path (relative or absolute)
    Returns:
        Resolved absolute path
    """
```

## Error Handling

**Validate boundaries, trust internals.** Validate aggressively at entry points (CLI, file I/O, config parsing). Once validated, trust data downstream.

Let errors propagate - catch at boundaries where you can handle meaningfully:

```python
# Good - propagate, catch at CLI
def run_pipeline(stages):
    return execute(build_dag(stages))  # May raise

# CLI catches
except StageNotFoundError as e:
    click.echo(f"Error: {e}", err=True)
```

**When to suppress vs propagate:**

| Condition | Action |
|-----------|--------|
| Unknown/invalid state | Propagate - fail fast |
| Invariant violation | Propagate - this is a bug |
| Cache miss, optional feature | Log and continue with fallback |
| Resource exhaustion | Propagate - architectural issue |

## Simplicity Over Abstraction

- **Don't create thin wrapper functions** - If it just calls one library function, inline it
- **Don't over-modularize** - A module with one public function used by one other module should be inlined
- **Three similar lines > premature abstraction** - Wait until the pattern is clear before extracting
- **No nested functions** - Use module-level for testability and fingerprinting

## Type Safety

- Zero tolerance for basedpyright warnings - resolve all errors AND warnings
- No blanket `# pyright: reportFoo=false` - use targeted ignores:
  ```python
  return json.load(f)  # type: ignore[return-value] - json returns Any
  ```
- Prefer type stubs (`pandas-stubs`, `types-PyYAML`) over ignores
- `Callable` over `Any` for functions; document why when using `Any`

## Python 3.13+ Types

- Empty collections: `list[int]()` not `: list[int] = []`
- Simplified Generator: `Generator[int]` not `Generator[int, None, None]`

## Comments

Prefer better code over comments. Add comments only for:

- Non-obvious WHY
- Timing constraints
- Known limitations

Never comment obvious WHAT (`# Add node` before `graph.add_node()`).

Write evergreen docs - avoid "recently added" or "as of version X".

## Enums Over Literals

For programmatic values, use enums (catches typos at type-check time):

```python
# Good
class OutputType(Enum):
    OUT = "out"
    METRIC = "metric"
    PLOT = "plot"

# Bad - typos not caught
output_type: Literal["out", "metric", "plot"]
```

## See Also

- [Testing Guide](testing.md) - Writing tests
- [CLI Development](cli.md) - CLI patterns
- [Common Gotchas](gotchas.md) - Pitfalls to avoid
