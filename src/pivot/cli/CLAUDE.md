# Pivot CLI - Development Guidelines

## The `pivot_command` Decorator (Critical)

**All new CLI commands MUST use `@cli_decorators.pivot_command()` instead of `@click.command()`.**

The `pivot_command` decorator wraps Click commands with two essential behaviors:

1. **Auto-discovery** - Automatically calls `discover_and_register()` before the command runs
2. **Error handling** - Converts `PivotError` exceptions to user-friendly `ClickException` messages

```python
from pivot.cli import decorators as cli_decorators

# Standard command - auto-discovers stages before running
@cli_decorators.pivot_command()
def list_cmd() -> None:
    """List registered stages."""
    # Registry is guaranteed to be populated here
    stages = registry.REGISTRY.list_stages()
    ...

# Command that doesn't need the registry
@cli_decorators.pivot_command(auto_discover=False)
def init() -> None:
    """Initialize new project."""
    # No discovery needed - creating new project
    ...
```

### When to use `auto_discover=False`

Set `auto_discover=False` only for commands that don't use the stage registry:

| Command | auto_discover | Reason |
|---------|---------------|--------|
| run, list, export | True (default) | Need registry to find stages |
| checkout, track | True (default) | Need registry for validation |
| init | False | Creates new project (no pipeline yet) |
| schema | False | Outputs JSON schema only |
| push, pull | False | Read from lock files, not registry |

### Why this matters

Without auto-discovery, commands fail if called before any other pivot command:

```bash
# In a fresh shell, this would fail without auto-discovery:
$ pivot list
No stages registered.  # Wrong! Should discover pivot.yaml first
```

The decorator ensures consistent behavior regardless of command order.

### For group subcommands

Commands under a `@click.group()` (like `pivot metrics show`) can't use `pivot_command`. Use `@with_error_handling` and call `ensure_stages_registered()` explicitly:

```python
from pivot.cli.run import ensure_stages_registered

@metrics.command("show")
@cli_decorators.with_error_handling
def metrics_show() -> None:
    ensure_stages_registered()  # Manual discovery for group subcommands
    ...
```

## Input Validation (Critical)

**Validate inputs as early as possible** - use Click's built-in validation in option/argument decorators.

### Numeric Options

Always use `click.IntRange` or `click.FloatRange` for numeric inputs:

```python
# Good - validates at argument parsing time
@click.option("--precision", type=click.IntRange(min=0), default=5)
@click.option("--jobs", type=click.IntRange(min=1), default=20)
@click.option("--debounce", type=click.IntRange(min=0), default=300)

# Bad - allows invalid values through
@click.option("--precision", type=int, default=5)  # Allows negative!
```

### Path Options

Use `click.Path` with appropriate parameters:

```python
@click.option("--output", type=click.Path(path_type=pathlib.Path))
@click.option("--config", type=click.Path(exists=True, dir_okay=False))
```

### Choice Options

Use `click.Choice` for limited valid values:

```python
@click.option("--format", type=click.Choice(["json", "yaml", "csv"]))
```

### Why Early Validation Matters

1. **Better error messages** - Click provides user-friendly error messages automatically
2. **Fail fast** - Don't waste time processing before discovering invalid input
3. **Type safety** - Validated inputs have correct types in the function body
4. **Consistency** - Users get the same error format for all validation failures