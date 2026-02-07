# CLI Consistency: Align Failure Defaults and Clarify JSONL Output

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pivot run` and `pivot repro` share the same default failure behavior (fail-fast) and expose `--jsonl` as the explicit streaming output flag with `--json` kept as an alias.

**Architecture:** Three changes: (1) flip `run`'s default from keep-going to fail-fast and give it `--keep-going`, (2) give `repro` a `--fail-fast` flag for symmetry, (3) add `--jsonl` flag to both commands with `--json` as alias. All validation logic (`_run_common.py`) already uses the `as_json` boolean so flag aliasing is transparent.

**Tech Stack:** Click (CLI framework), pytest, basedpyright, ruff

---

## Summary of Current State

| Aspect | `pivot run` | `pivot repro` |
|--------|-------------|---------------|
| Default error mode | keep-going | fail-fast |
| Has `--fail-fast` | Yes | No |
| Has `--keep-going` / `-k` | No | Yes |
| JSON flag | `--json` (help says "JSON") | `--json` (help says "JSON") |

## Target State

| Aspect | `pivot run` | `pivot repro` |
|--------|-------------|---------------|
| Default error mode | **fail-fast** | fail-fast |
| Has `--fail-fast` | Yes | **Yes** |
| Has `--keep-going` / `-k` | **Yes** | Yes |
| JSON flag | **`--jsonl` / `--json`** (help says JSONL) | **`--jsonl` / `--json`** (help says JSONL) |

---

### Task 1: Add `--keep-going` to `run` and `--fail-fast` to `repro` with fail-fast default for both

**Files:**
- Modify: `src/pivot/cli/run.py` (lines 322-415)
- Modify: `src/pivot/cli/repro.py` (lines 741-896)
- Modify: `tests/cli/test_run.py` (lines 172-188, 459-471)
- Modify: `tests/cli/test_repro.py` (lines 390-403)
- Modify: `tests/cli/test_cli_run_keep_going.py` (lines 1-334)

**Step 1: Update `run.py` — add `--keep-going` flag, change default to fail-fast**

In `src/pivot/cli/run.py`, replace the `--fail-fast` option decorator and add a `--keep-going` option. Make them mutually exclusive:

Replace the existing `--fail-fast` click option (lines 357-361):
```python
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure (default: keep going)",
)
```

With both flags:
```python
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure (default).",
)
@click.option(
    "--keep-going",
    "-k",
    is_flag=True,
    help="Continue running stages after failures.",
)
```

Update the `run` function signature to accept `keep_going: bool`:
```python
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    force: bool,
    tui_flag: bool,
    as_json: bool,
    show_output: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    fail_fast: bool,
    keep_going: bool,         # <-- add this
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> None:
```

Add mutual exclusion validation after existing validations (around line 409):
```python
    # Validate --fail-fast and --keep-going are mutually exclusive
    if fail_fast and keep_going:
        raise click.ClickException("--fail-fast and --keep-going are mutually exclusive")
```

Change the default from keep-going to fail-fast (line 415). Replace:
```python
    # Default: keep going; --fail-fast stops on first failure
    on_error = OnError.FAIL if fail_fast else OnError.KEEP_GOING
```
With:
```python
    # Default: fail-fast; --keep-going continues after failures
    on_error = OnError.KEEP_GOING if keep_going else OnError.FAIL
```

**Step 2: Update `repro.py` — add `--fail-fast` flag for symmetry**

In `src/pivot/cli/repro.py`, add a `--fail-fast` option alongside the existing `--keep-going`. Replace the existing `--keep-going` option (lines 792-797):
```python
@click.option(
    "--keep-going",
    "-k",
    is_flag=True,
    help="Continue running stages after failures; skip only downstream dependents.",
)
```

With both flags:
```python
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure (default).",
)
@click.option(
    "--keep-going",
    "-k",
    is_flag=True,
    help="Continue running stages after failures; skip only downstream dependents.",
)
```

Update the `repro` function signature to accept `fail_fast: bool`:
```python
def repro(
    ctx: click.Context,
    stages: tuple[str, ...],
    dry_run: bool,
    explain: bool,
    force: bool,
    watch: bool,
    debounce: int | None,
    tui_flag: bool,
    as_json: bool,
    show_output: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    fail_fast: bool,          # <-- add this
    keep_going: bool,
    serve: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
    allow_missing: bool,
) -> None:
```

Add mutual exclusion validation (after `--allow-missing` validation, around line 878):
```python
    # Validate --fail-fast and --keep-going are mutually exclusive
    if fail_fast and keep_going:
        raise click.ClickException("--fail-fast and --keep-going are mutually exclusive")
```

The existing on_error line (line 896) already handles the logic correctly:
```python
    on_error = OnError.KEEP_GOING if keep_going else OnError.FAIL
```
This is correct — default is fail-fast (`OnError.FAIL`), `--keep-going` switches to `OnError.KEEP_GOING`.

**Step 3: Update tests — fix tests that assert old defaults or missing options**

In `tests/cli/test_run.py`:

1. **Remove `test_run_does_not_have_keep_going_option`** (lines 459-471) — `run` now HAS `--keep-going`.

2. **Update `test_run_default_keeps_going_continues_after_failure`** (lines 172-188) — rename and invert behavior. The test should now verify fail-fast is the default. Replace with:
```python
def test_run_default_fails_fast(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run defaults to fail-fast mode - stops on first failure."""
    register_test_stage(_helper_failing_stage, name="failing")
    register_test_stage(_helper_stage_a, name="stage_a")

    # Run failing then stage_a - default fail-fast should stop
    result = runner.invoke(cli.cli, ["run", "failing", "stage_a"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
```

In `tests/cli/test_cli_run_keep_going.py`:

1. **Update module docstring** (lines 1-4) to:
```python
"""Tests for keep-going / fail-fast behavior in CLI commands.

Both run and repro default to fail-fast.
Use --keep-going / -k to continue after failures.
"""
```

2. **Update `test_run_default_keeps_going`** (lines 288-306) — rename and change to verify new default. Replace with:
```python
def test_run_default_fails_fast(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run defaults to fail-fast mode (stops on first failure)."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    # Run both stages - default should be fail-fast
    result = runner.invoke(cli.cli, ["run", "failing", "succeeding"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
```

3. **Add a test for `run --keep-going`**:
```python
def test_run_keep_going_continues_after_failure(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --keep-going continues after failures."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    result = runner.invoke(cli.cli, ["run", "--keep-going", "failing", "succeeding"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
    assert "succeeding: done" in result.output
    assert (tmp_path / "succeeding.txt").read_text() == "success"
```

4. **Add a test for `run -k` short flag**:
```python
def test_run_keep_going_short_flag(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run -k short flag works the same as --keep-going."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    result = runner.invoke(cli.cli, ["run", "-k", "failing", "succeeding"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
    assert "succeeding: done" in result.output
```

5. **Add a test for `run --keep-going` shown in help**:
```python
def test_run_keep_going_flag_shown_in_help(runner: CliRunner) -> None:
    """run --keep-going flag is documented in help."""
    result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--keep-going" in result.output
    assert "-k" in result.output
```

6. **Add a test for `repro --fail-fast` shown in help**:
```python
def test_repro_fail_fast_flag_shown_in_help(runner: CliRunner) -> None:
    """repro --fail-fast flag is documented in help."""
    result = runner.invoke(cli.cli, ["repro", "--help"])

    assert result.exit_code == 0
    assert "--fail-fast" in result.output
```

7. **Add mutual exclusion tests**:
```python
def test_run_fail_fast_and_keep_going_mutually_exclusive(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --fail-fast and --keep-going are mutually exclusive."""
    register_test_stage(_stage_process, name="process")
    (tmp_path / "input.txt").write_text("data")

    result = runner.invoke(cli.cli, ["run", "--fail-fast", "--keep-going", "process"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_repro_fail_fast_and_keep_going_mutually_exclusive(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --fail-fast and --keep-going are mutually exclusive."""
    register_test_stage(_stage_process, name="process")
    (tmp_path / "input.txt").write_text("data")

    result = runner.invoke(cli.cli, ["repro", "--fail-fast", "--keep-going"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()
```

**Step 4: Run tests**

```bash
cd /home/sami/pivot/roadmap-385
uv run pytest tests/cli/test_run.py tests/cli/test_repro.py tests/cli/test_cli_run_keep_going.py -v
```

Expected: All tests pass.

---

### Task 2: Add `--jsonl` flag as primary, keep `--json` as alias

**Files:**
- Modify: `src/pivot/cli/run.py` (line 336)
- Modify: `src/pivot/cli/repro.py` (line 771)
- Modify: `tests/cli/test_run.py`
- Modify: `tests/cli/test_repro.py`
- Modify: `tests/cli/test_cli_run_keep_going.py`

**Step 1: Update `run.py` — replace `--json` option with `--jsonl`/`--json` alias**

In `src/pivot/cli/run.py`, replace the `--json` option (line 336):
```python
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
```

With:
```python
@click.option(
    "--jsonl",
    "--json",
    "as_json",
    is_flag=True,
    help="Stream results as JSONL (one JSON object per line).",
)
```

Note: Click's first option name becomes the canonical one. Both `--jsonl` and `--json` map to the same `as_json` parameter. No function signature change needed.

**Step 2: Update `repro.py` — same change**

In `src/pivot/cli/repro.py`, replace the `--json` option (line 771):
```python
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
```

With:
```python
@click.option(
    "--jsonl",
    "--json",
    "as_json",
    is_flag=True,
    help="Stream results as JSONL (one JSON object per line).",
)
```

**Step 3: Update the `validate_show_output` error message in `_run_common.py`**

In `src/pivot/cli/_run_common.py`, update the error message at line 152:
```python
    if show_output and as_json:
        raise click.ClickException("--show-output and --json are mutually exclusive")
```
To:
```python
    if show_output and as_json:
        raise click.ClickException("--show-output and --jsonl are mutually exclusive")
```

Also update the `validate_tui_log` error message at line 132:
```python
    if as_json:
        raise click.ClickException("--tui-log cannot be used with --json")
```
To:
```python
    if as_json:
        raise click.ClickException("--tui-log cannot be used with --jsonl")
```

Also update the `validate_tui_log` error message in `repro.py`'s `--tui and --json` validation (line 860 in `repro.py` and line 406 in `run.py`):
```python
        raise click.ClickException("--tui and --json are mutually exclusive")
```
To:
```python
        raise click.ClickException("--tui and --jsonl are mutually exclusive")
```

**Step 4: Add tests for `--jsonl` flag**

In `tests/cli/test_run.py`, add:
```python
def test_run_jsonl_flag_accepted(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--jsonl flag works and streams JSONL output."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--jsonl", "stage_a"])

    assert result.exit_code == 0
    lines = [ln for ln in result.output.strip().split("\n") if ln]
    assert len(lines) > 0
    for line in lines:
        json.loads(line)  # Should not raise
```

In `tests/cli/test_repro.py`, add:
```python
def test_repro_jsonl_flag_accepted(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--jsonl flag works and streams JSONL output."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "--jsonl"])

    assert result.exit_code == 0
    lines = [ln for ln in result.output.strip().split("\n") if ln]
    assert len(lines) > 0
    for line in lines:
        json.loads(line)  # Should not raise
```

**Step 5: Add help text tests for JSONL wording**

In `tests/cli/test_cli_run_keep_going.py` (or a new section in test_run.py — add in test_run.py):
```python
def test_run_help_shows_jsonl(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """run help shows --jsonl flag with JSONL description."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--jsonl" in result.output
    assert "JSONL" in result.output


def test_repro_help_shows_jsonl(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """repro help shows --jsonl flag with JSONL description."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["repro", "--help"])

    assert result.exit_code == 0
    assert "--jsonl" in result.output
    assert "JSONL" in result.output
```

**Step 6: Update existing tests that reference `--json` in error messages**

In `tests/cli/test_run.py`, update `test_run_tui_log_cannot_use_with_json` (line 241):
Change assertion from `assert "--json" in result.output` to `assert "--jsonl" in result.output`.

Update `test_run_tui_and_json_mutually_exclusive` (line 258):
Change assertion from `assert "mutually exclusive" in result.output.lower()` — this should still pass since the error message still contains "mutually exclusive".

Update `test_run_show_output_mutually_exclusive_with_json` (line 531):
Change assertion from `assert "--show-output and --json are mutually exclusive" in result.output` to `assert "--show-output and --jsonl are mutually exclusive" in result.output`.

In `tests/cli/test_repro.py`, update `test_repro_tui_log_cannot_use_with_json` (line 316):
Change assertion from `assert "--json" in result.output` to `assert "--jsonl" in result.output`.

Update `test_repro_show_output_mutually_exclusive_with_json` (line 523):
Change assertion from `assert "--show-output and --json are mutually exclusive" in result.output` to `assert "--show-output and --jsonl are mutually exclusive" in result.output`.

**Step 7: Run all tests**

```bash
cd /home/sami/pivot/roadmap-385
uv run pytest tests/cli/test_run.py tests/cli/test_repro.py tests/cli/test_cli_run_keep_going.py tests/cli/test_cli_run_common.py tests/cli/test_jsonl_sink.py -v
```

Expected: All tests pass.

---

### Task 3: Quality checks and final verification

**Step 1: Run full test suite**

```bash
cd /home/sami/pivot/roadmap-385
uv run pytest tests/ -n auto
```

Expected: All tests pass.

**Step 2: Run quality checks**

```bash
cd /home/sami/pivot/roadmap-385
uv run ruff format . && uv run ruff check . && uv run basedpyright
```

Expected: Clean output.

**Step 3: Verify help text manually**

```bash
cd /home/sami/pivot/roadmap-385
uv run pivot run --help
uv run pivot repro --help
```

Verify:
- Both show `--fail-fast` and `--keep-going` / `-k`
- Both show `--jsonl` / `--json` with JSONL description
- Both default to fail-fast (check help text says "default" next to `--fail-fast`)

**Step 4: Create bookmark**

```bash
cd /home/sami/pivot/roadmap-385
jj bookmark create issue-385 -r @
```
