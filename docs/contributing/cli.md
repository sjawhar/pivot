# CLI Development

Guidelines for developing Pivot CLI commands.

## Quick Checklist

Before submitting a PR with a new CLI command:

- [ ] File created at `src/pivot/cli/<command>.py`
- [ ] Uses `@cli_decorators.pivot_command()` decorator
- [ ] Registered in `_LAZY_COMMANDS` (and `COMMAND_CATEGORIES`) in `src/pivot/cli/__init__.py`
- [ ] Has shell completion for stage arguments
- [ ] Validates inputs using Click types
- [ ] Shows explicit message for empty states
- [ ] Integration test at `tests/integration/test_cli_<command>.py`
- [ ] Tests cover success path, error paths, and output formats

## The `pivot_command` Decorator

**All new CLI commands MUST use `@cli_decorators.pivot_command()` instead of `@click.command()`.**

This decorator provides:

1. **Auto-discovery** - Discovers `pipeline.py` and stores the Pipeline in the Click context before the command runs
2. **Error handling** - Converts `PivotError` exceptions to user-friendly `ClickException` messages with their suggestion. Any other exception becomes one line from `exceptions.describe_error`, never a Python repr: an `OSError` as `<strerror>: <path>` (`<strerror>: <path> -> <path2>` for two paths, `<strerror>` for none), anything else as `Unexpected <type>: <message>`; `pivot --verbose` logs its traceback. Let an unexpected exception reach this handler rather than printing `str(e)` or `repr(e)` yourself; `ctx.exit(code)` passes through with its exit code.

`allow_all=True` adds a `--all` flag that discovers every pipeline in the project.

```python
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers

# Standard command - auto-discovers stages before running
@cli_decorators.pivot_command()
def list_cmd() -> None:
    """List registered stages."""
    # Pipeline is guaranteed to be in context here
    stages = cli_helpers.list_stages()
    ...

# Command that doesn't need the pipeline
@cli_decorators.pivot_command(auto_discover=False)
def init() -> None:
    """Initialize new project."""
    # No discovery needed - creating new project
    ...
```

### When to Use `auto_discover=False`

Set `auto_discover=False` only for commands that don't read stage metadata, or that must keep working when `pipeline.py` is broken and discover the pipeline themselves when they need it. The pipeline is the canonical source of stage metadata: a command that inspects outputs, tags or deps uses auto-discovery.

| Command | auto_discover | Reason |
|---------|---------------|--------|
| run, repro, commit, verify | True (default), allow_all=True | Need the pipeline to find stages |
| list, dag, status | True (default), allow_all=True | Need the pipeline; `status` skips its stages section when there is none |
| checkout | True (default), allow_all=True | Needs the pipeline; handles a missing one |
| push, pull, fetch | True (default), allow_all=True | Need the pipeline for output cache filtering |
| track | False | Discovers every pipeline itself (`discovery.declared_outputs`) and refuses a path that overlaps any of their outputs, or a `.pvt` sidecar that the lock of a removed stage records as that stage's file |
| get | False | Restores files and `.pvt` targets from git; discovers the pipeline itself for stage targets, so a broken `pipeline.py` stops only stage targets |
| gc | False | Reads locks, `.pvt` files, run-cache rows and links on disk, so a broken `pipeline.py` never stops a user from freeing disk |
| import, update | False | Work from git revisions, lock files and `.pvt` files |
| history, show | False | Read run history only |
| init, doctor, check-ignore, forget-cache | False | No stage data needed |

### Group Subcommands

Commands under a `@click.group()` (like `pivot metrics show`) can't use `pivot_command`. Use `@with_error_handling` and call `ensure_stages_registered()` explicitly:

```python
from pivot.cli._run_common import ensure_stages_registered

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
    from pivot.cli import helpers as cli_helpers

    all_stages = cli_helpers.list_stages()

    if not all_stages:
        click.echo("No stages registered.")
        return

    if stages:
        all_stages = [s for s in all_stages if s in stages]

    if output_json:
        import json
        result = {"stages": all_stages}
        click.echo(json.dumps(result, indent=2))
    else:
        for stage in all_stages:
            click.echo(stage)
```

### 2. Register the Command

Commands load lazily: `pivot --help` imports no command module, and `pivot <command> --help` imports only that command's. Add an entry to `_LAZY_COMMANDS` (command name, module, attribute, one-line help) and put the name in a `COMMAND_CATEGORIES` section:

```python
# src/pivot/cli/__init__.py
_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    ...
    "mycommand": ("pivot.cli.mycommand", "mycommand", "One-line description."),
}
```

Import pandas, numpy, loky, the engine or the executor worker inside the command function, not at module top; `tests/cli/test_cli_performance.py` fails a command whose `--help` loads them.

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

Tests run the command through Click's `CliRunner` against a temporary project with a real `pipeline.py`. Use the `pipeline_dir` fixture from `tests/conftest.py`: it creates `.pivot/`, changes into the directory, and resets the cached project root (`project._project_root_cache` survives between tests in one process). Shown here for `pivot list`; substitute your command:

```python
# tests/integration/test_cli_mycommand.py
import pathlib

from click.testing import CliRunner

from pivot.cli import cli

PIPELINE = """
import pandas as pd

from pivot import Pipeline, stage


@stage
def process(data: pd.DataFrame) -> pd.DataFrame:
    return data


with Pipeline("demo") as pipeline:
    process(pipeline.input("data.csv"))
"""


def test_mycommand_no_pipeline(pipeline_dir: pathlib.Path) -> None:
    """Without pipeline.py the command explains what is missing."""
    result = CliRunner().invoke(cli, ["list"])

    assert "pipeline.py" in result.output


def test_mycommand_with_stages(pipeline_dir: pathlib.Path) -> None:
    """With a pipeline the command lists its stages."""
    (pipeline_dir / "pipeline.py").write_text(PIPELINE)
    (pipeline_dir / "data" / "raw").mkdir(parents=True)
    (pipeline_dir / "data" / "raw" / "data.csv").write_text("a,b\n1,2\n")

    result = CliRunner().invoke(cli, ["list"])

    assert result.exit_code == 0
    assert "demo/process" in result.output
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
