# Pivot CLI

Click commands behind the `pivot` entry point. Read `docs/contributing/cli.md` before you add or change a
command: it covers `pivot_command`, `auto_discover`, registration in `_LAZY_COMMANDS` and `COMMAND_CATEGORIES`,
option validation, error output and a worked example.

- Commands load lazily: a new command imports pandas, numpy, loky, the engine and the executor worker inside its
  function, not at module top. `packages/pivot/tests/cli/test_cli_performance.py` checks `--help` for every command
  except those listed in its `_HEAVY_HELP_COMMANDS`.
- Every command gets an integration test (`packages/pivot/tests/AGENTS.md`).
