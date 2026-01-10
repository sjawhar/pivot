# Contributing

Guide for contributing to Pivot development.

## Development Setup

### Using Dev Container (Recommended)

The repository includes a dev container configuration in `.devcontainer/`:

1. Open the project in VS Code
2. Install the "Dev Containers" extension
3. Click "Reopen in Container" when prompted

### Manual Setup

```bash
# Clone the repository
git clone https://github.com/sjawhar/pivot.git
cd pivot

# Install dependencies with uv
uv sync --active
```

## Quality Commands

Run these before submitting changes:

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run basedpyright .

# Run tests (parallel)
uv run pytest tests/ -n auto
```

All four must pass before merging.

## Test Structure

```
tests/
├── unit/           # Unit tests per module
├── integration/    # Full pipeline tests
└── fingerprint/    # Code change detection tests
```

### Writing Tests

Tests use pytest with the `tmp_path` fixture for isolation:

```python
def test_stage_execution(tmp_path: pathlib.Path) -> None:
    """Test that a stage runs correctly."""
    # Setup
    input_file = tmp_path / "input.csv"
    input_file.write_text("a,b\n1,2\n")

    # Exercise
    result = run_stage(...)

    # Verify
    assert result.status == StageStatus.SUCCESS
```

### Cross-Process Tests

When testing multiprocessing behavior, use file-based state instead of shared memory:

```python
# Bad - shared mutable state silently fails in multiprocessing
execution_log = list[str]()

def my_stage():
    execution_log.append("ran")  # Each process has its own copy!

# Good - file-based logging for cross-process communication
def my_stage():
    with open("log.txt", "a") as f:
        f.write("ran\n")
```

This is relevant both for contributors writing tests and users writing their own pipeline tests.

## Code Style

### Import Style (Google)

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

### TypedDict Constructor Syntax

Always use constructor syntax for TypedDicts:

```python
class Result(TypedDict):
    status: str
    value: int

# Good
return Result(status="ok", value=42)

# Bad - no type validation
return {"status": "ok", "value": 42}
```

### Early Returns

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

### Private Functions

Use underscore prefix for module-internal helpers:

```python
def public_function():
    """Public API."""
    return _internal_helper()

def _internal_helper():
    """Not part of public API."""
    pass
```

## Pull Request Guidelines

Use the PR template at `.github/pull_request_template.md`:

1. **Overview** - What does this PR do?
2. **Issue Link** - Closes #123
3. **Approach** - How you solved it and alternatives considered
4. **Testing** - How to verify it works
5. **Checklist** - Tests pass, docs updated, etc.

Highlight areas needing careful review (complex logic, edge cases, design decisions).

## Architecture Overview

Key design decisions:

- **Per-stage lock files** - O(n) writes vs O(n²) with monolithic manifest
- **AST fingerprinting** - Automatic code change detection via `getclosurevars()`
- **loky reusable executor** - Warm workers persist across calls
- **LMDB state database** - All persistent state uses key prefixes

See [Architecture Overview](architecture/overview.md) for details.
