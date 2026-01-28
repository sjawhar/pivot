# CLI Development

Guidelines for developing Pivot CLI commands.

## Quick Checklist

Before submitting a PR with a new CLI command:

- [ ] File created at `src/pivot/cli/<command>.py`
- [ ] Uses `@cli_decorators.pivot_command()` decorator
- [ ] Registered in `src/pivot/cli/__init__.py`
- [ ] Has shell completion for stage arguments
- [ ] Validates inputs using Click types
- [ ] Shows explicit message for empty states
- [ ] Integration test at `tests/integration/test_cli_<command>.py`
- [ ] Tests cover success path, error paths, and output formats

## The `pivot_command` Decorator

**All new CLI commands MUST use `@cli_decorators.pivot_command()` instead of `@click.command()`.**

This decorator provides:

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

### When to Use `auto_discover=False`

Set `auto_discover=False` only for commands that don't use the stage registry:

| Command | auto_discover | Reason |
|---------|---------------|--------|
| run, list, export | True (default) | Need registry to find stages |
| checkout, track | True (default) | Need registry for validation |
| init | False | Creates new project (no pipeline yet) |
| schema | False | Outputs JSON schema only |
| push, pull | False | Read from lock files, not registry |

### Group Subcommands

Commands under a `@click.group()` (like `pivot metrics show`) can't use `pivot_command`. Use `@with_error_handling` and call `ensure_stages_registered()` explicitly:

```python
from pivot.cli.run import ensure_stages_registered

@metrics.command("show")
@cli_decorators.with_error_handling
def metrics_show() -> None:
    ensure_stages_registered()  # Manual discovery for group subcommands
    ...
```

## Adding a New Command

### 1. Create the Command File

```python
# src/pivot/cli/mycommand.py
import click

from pivot.cli import decorators as cli_decorators


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def mycommand(stages: tuple[str, ...], verbose: bool, output_json: bool) -> None:
    """One-line description of what this command does."""
    from pivot import registry

    all_stages = registry.REGISTRY.list_stages()

    if not all_stages:
        click.echo("No stages registered.")
        return

    if stages:
        all_stages = [s for s in all_stages if s.name in stages]

    if output_json:
        import json
        result = {"stages": [s.name for s in all_stages]}
        click.echo(json.dumps(result, indent=2))
    else:
        for stage in all_stages:
            click.echo(stage.name)
```

### 2. Register the Command

```python
# src/pivot/cli/__init__.py
from pivot.cli.mycommand import mycommand

# ... at the bottom with other commands ...
cli.add_command(mycommand)
```

### 3. Add Shell Completion

```python
from pivot.cli import completion

@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
def mycommand(stages: tuple[str, ...]) -> None:
    ...
```

Available completers:

- `completion.complete_stages` - Stage names
- `completion.complete_targets` - Files and stage names

### 4. Write Integration Tests

```python
# tests/integration/test_cli_mycommand.py
import pathlib

from click.testing import CliRunner

from pivot.cli import cli


def test_mycommand_no_stages(tmp_path: pathlib.Path) -> None:
    """Test mycommand with empty pipeline."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path("pivot.yaml").write_text("stages: {}")

        result = runner.invoke(cli, ["mycommand"])

        assert result.exit_code == 0
        assert "No stages registered" in result.output


def test_mycommand_with_stages(tmp_path: pathlib.Path) -> None:
    """Test mycommand lists stages."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path("pivot.yaml").write_text("""
stages:
  process:
    python: stages.process
    deps:
      data: data.csv
    outs:
      output: output.csv
""")
        pathlib.Path("stages.py").write_text("""
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class ProcessOutputs(TypedDict):
    output: Annotated[str, outputs.Out("output.csv", loaders.PathOnly())]

def process(
    data: Annotated[str, outputs.Dep("data.csv", loaders.PathOnly())],
) -> ProcessOutputs:
    return {"output": "output.csv"}
""")
        pathlib.Path("data.csv").write_text("a,b\n1,2\n")

        result = runner.invoke(cli, ["mycommand"])

        assert result.exit_code == 0
        assert "process" in result.output
```

## Input Validation

**Validate inputs as early as possible** - Use Click's built-in validation.

### Numeric Options

Always use `click.IntRange` or `click.FloatRange`:

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

### Mutually Exclusive Options

```python
@click.option("--verbose", is_flag=True)
@click.option("--quiet", is_flag=True)
def cmd(verbose: bool, quiet: bool) -> None:
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive")
```

## Output Conventions

### Explicit Output

Always show messages for empty states:

```python
# Good
if not stages:
    click.echo("No stages registered.")
    return

# Bad - silent empty output
for stage in stages:
    click.echo(stage.name)
```

### JSON Output

JSON output must include all requested keys (empty arrays, not omitted):

```python
# Good
result = {"stages": [], "errors": []}

# Bad - omits empty keys
result = {}
if stages:
    result["stages"] = stages
```

### Multiple Output Formats

Support multiple output formats when useful:

```python
@click.option("--format", type=click.Choice(["text", "json", "md"]), default="text")
def mycommand(format: str) -> None:
    match format:
        case "json":
            click.echo(json.dumps(data, indent=2))
        case "md":
            click.echo(format_markdown(data))
        case "text":
            click.echo(format_text(data))
```

### Progress Indicators

For long-running operations:

```python
import click

with click.progressbar(items, label="Processing") as bar:
    for item in bar:
        process(item)
```

## See Also

- [Getting Started](setup.md) - Development environment
- [Code Style](style.md) - Coding conventions
- [Testing Guide](testing.md) - Writing tests
