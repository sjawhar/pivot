# Pivot CLI

Click commands behind the `pivot` entry point. A worked example, output conventions and which
commands discover the pipeline: `docs/contributing/cli.md`.

## Adding a command

- Put it in `cli/<name>.py` and decorate it with `@cli_decorators.pivot_command()`, not
  `@click.command()`. Register it in `_LAZY_COMMANDS` in `cli/__init__.py`, and in
  `COMMAND_CATEGORIES` for its `pivot --help` section.
- Commands load lazily. Import pandas, numpy, loky, the engine or the executor worker inside the
  function, not at module top (`tests/cli/test_cli_performance.py` checks each `--help`).
- `pivot_command` discovers `pipeline.py` into the Click context before the command runs.
  Set `auto_discover=False` for a command that reads no stage metadata, or that must still work
  when `pipeline.py` is broken and discovers only when it needs to (`gc`, `get`, `track`).
  `allow_all=True` adds `--all` (every pipeline in the project).
- A subcommand of a `@click.group()` (`pivot metrics show`) cannot use `pivot_command`: decorate
  it with `@cli_decorators.with_error_handling` and call `_run_common.ensure_stages_registered()`
  first.
- Validate options with Click types (`click.IntRange`, `click.Path`, `click.Choice`), not in the
  function body.

## Errors

- Raise a `PivotError` subclass from `pivot.exceptions`; the decorator prints its message and
  suggestion.
- Let any other exception reach the decorator, which prints one line from
  `exceptions.describe_error` and logs the traceback under `pivot --verbose`. Do not print
  `str(e)` or `repr(e)` yourself. `ctx.exit(code)` passes through with its code.

## Tests

Every command gets an integration test that runs it on real files for its success path, its
error paths and each output format (`packages/pivot/tests/AGENTS.md`).
