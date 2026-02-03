# Pull/Fetch CLI Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pivot pull` match git/dvc semantics (fetch + checkout), add `fetch` command, promote `data` subcommands to top-level, and reorganize CLI categories.

**Architecture:** Rename current `pull` to `fetch`, create new `pull` that chains fetch + checkout. Move `data diff` and `data get` to top-level commands. Update CLI registry and categories.

**Tech Stack:** Python 3.13, Click, pytest

**Issue:** https://github.com/sjawhar/pivot/issues/336

---

## Task 1: Implement All CLI Changes

**Files:**
- Modify: `src/pivot/cli/remote.py`
- Modify: `src/pivot/cli/data.py`
- Modify: `src/pivot/cli/__init__.py`

### Step 1: Rename `pull` to `fetch` in remote.py

In `src/pivot/cli/remote.py`, rename the `pull` function to `fetch` and update docstrings/messages:

```python
@cli_decorators.pivot_command(auto_discover=False)
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option("-r", "--remote", "remote_name", help="Remote name (uses default if not specified)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be fetched")
@click.option(
    "-j", "--jobs", type=click.IntRange(min=1), default=None, help="Parallel download jobs"
)
@click.pass_context
def fetch(
    ctx: click.Context,
    targets: tuple[str, ...],
    remote_name: str | None,
    dry_run: bool,
    jobs: int | None,
) -> None:
    """Fetch cached outputs from remote storage to local cache.

    TARGETS can be stage names or file paths. If specified, fetches those
    outputs (and dependencies for stages). Otherwise, fetches all available
    files from remote.

    This command only downloads to the local cache. Use 'pivot pull' to also
    restore files to your workspace, or 'pivot checkout' to restore from cache.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    cache_dir = config.get_cache_dir()
    state_dir = config.get_state_dir()
    s3_remote, resolved_name = transfer.create_remote_from_name(remote_name)

    targets_list = list(targets) if targets else None

    if dry_run:
        if targets_list:
            needed = transfer.get_target_hashes(targets_list, state_dir, include_deps=True)
        else:
            needed = asyncio.run(s3_remote.list_hashes())

        local = transfer.get_local_cache_hashes(cache_dir)
        missing = needed - local
        if not quiet:
            click.echo(f"Would fetch {len(missing)} file(s) from '{resolved_name}'")
        return

    with state.StateDB(config.get_state_db_path()) as state_db:
        result = transfer.pull(
            cache_dir,
            state_dir,
            s3_remote,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            None if quiet else cli_helpers.make_progress_callback("Downloaded"),
        )

    if not quiet:
        transferred = result["transferred"]
        skipped = result["skipped"]
        failed = result["failed"]
        click.echo(
            f"Fetched from '{resolved_name}': {transferred} transferred, {skipped} skipped, {failed} failed"
        )

    cli_helpers.print_transfer_errors(result["errors"])
    if result["failed"] > 0:
        raise SystemExit(1)
```

### Step 2: Add new `pull` command in remote.py

Add after the `fetch` function. Note: we call the underlying functions directly (not `ctx.invoke`) to properly handle errors and exit codes:

```python
@cli_decorators.pivot_command(auto_discover=False)
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option("-r", "--remote", "remote_name", help="Remote name (uses default if not specified)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be pulled")
@click.option(
    "-j", "--jobs", type=click.IntRange(min=1), default=None, help="Parallel download jobs"
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing workspace files")
@click.option(
    "--only-missing",
    is_flag=True,
    help="Only restore files that don't exist in workspace",
)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.pass_context
def pull(
    ctx: click.Context,
    targets: tuple[str, ...],
    remote_name: str | None,
    dry_run: bool,
    jobs: int | None,
    force: bool,
    only_missing: bool,
    checkout_mode: str | None,
) -> None:
    """Pull cached outputs from remote and restore to workspace.

    Combines 'fetch' (download from remote) and 'checkout' (restore to workspace).
    This matches the behavior of 'git pull' and 'dvc pull'.

    TARGETS can be stage names or file paths. If specified, pulls those
    outputs (and dependencies for stages). Otherwise, pulls all available files.
    """
    if force and only_missing:
        raise click.ClickException("--force and --only-missing are mutually exclusive")

    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    cache_dir = config.get_cache_dir()
    state_dir = config.get_state_dir()
    s3_remote, resolved_name = transfer.create_remote_from_name(remote_name)

    targets_list = list(targets) if targets else None

    # Dry-run: show what would be fetched, don't proceed to checkout
    if dry_run:
        if targets_list:
            needed = transfer.get_target_hashes(targets_list, state_dir, include_deps=True)
        else:
            needed = asyncio.run(s3_remote.list_hashes())

        local = transfer.get_local_cache_hashes(cache_dir)
        missing = needed - local
        if not quiet:
            click.echo(f"Would fetch {len(missing)} file(s) from '{resolved_name}'")
        return

    # Step 1: Fetch from remote to cache
    with state.StateDB(config.get_state_db_path()) as state_db:
        fetch_result = transfer.pull(
            cache_dir,
            state_dir,
            s3_remote,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            None if quiet else cli_helpers.make_progress_callback("Downloaded"),
        )

    if not quiet:
        click.echo(
            f"Fetched from '{resolved_name}': {fetch_result['transferred']} transferred, "
            f"{fetch_result['skipped']} skipped, {fetch_result['failed']} failed"
        )

    cli_helpers.print_transfer_errors(fetch_result["errors"])

    # If fetch had failures, exit without checkout
    if fetch_result["failed"] > 0:
        raise SystemExit(1)

    # Step 2: Checkout from cache to workspace
    # Import here to avoid circular imports at module level
    from pivot.cli import checkout as checkout_mod

    ctx.invoke(
        checkout_mod.checkout,
        targets=targets,
        checkout_mode=checkout_mode,
        force=force,
        only_missing=only_missing,
    )
```

### Step 3: Promote `data diff` to top-level `diff` in data.py

In `src/pivot/cli/data.py`, change `data_diff` to a standalone `diff` command:

```python
@cli_decorators.pivot_command()
@click.argument("targets", nargs=-1, required=True)
@click.option("--key", "key_cols", help="Comma-separated key columns for row matching")
@click.option("--positional", is_flag=True, help="Use positional (row-by-row) matching")
@click.option("--summary", is_flag=True, help="Show summary only (schema + counts)")
@click.option("--no-tui", is_flag=True, help="Print to stdout instead of launching TUI")
@click.option(
    "--json",
    "output_format",
    flag_value=OutputFormat.JSON,
    help="Output as JSON (implies --no-tui)",
)
@click.option(
    "--md",
    "output_format",
    flag_value=OutputFormat.MD,
    help="Output as Markdown (implies --no-tui)",
)
@click.option(
    "--max-rows", default=None, type=click.IntRange(min=1), help="Max rows for comparison"
)
def diff(
    targets: tuple[str, ...],
    key_cols: str | None,
    positional: bool,
    summary: bool,
    no_tui: bool,
    output_format: OutputFormat | None,
    max_rows: int | None,
) -> None:
    """Compare data files in workspace against git HEAD.

    Compares CSV, JSON, and JSONL files showing schema changes, row additions,
    deletions, and modifications. Detects reorder-only changes.
    """
    # Function body unchanged from data_diff
```

Remove the `@data.command("diff")` and `@cli_decorators.with_error_handling` decorators (pivot_command includes error handling).

### Step 4: Promote `data get` to top-level `get` in data.py

```python
@cli_decorators.pivot_command(auto_discover=False)
@click.argument("targets", nargs=-1, required=True)
@click.option(
    "--rev",
    "-r",
    required=True,
    help="Git revision (SHA, branch, tag) to retrieve files from",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output path for single file target (incompatible with multiple targets or stage names)",
)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def get(
    targets: tuple[str, ...],
    rev: str,
    output: pathlib.Path | None,
    checkout_mode: str | None,
    force: bool,
) -> None:
    """Retrieve files or stage outputs from a specific git revision.

    TARGETS can be file paths or stage names.

    \b
    Examples:
      pivot get --rev v1.0 model.pkl              # Get file from tag
      pivot get --rev v1.0 model.pkl -o old.pkl   # Get file to alternate location
      pivot get --rev abc123 train                # Get all outputs from stage
    """
    # Function body unchanged from data_get
```

### Step 5: Remove the `data` command group in data.py

Delete these lines from `src/pivot/cli/data.py`:

```python
@click.group()
def data() -> None:
    """Inspect and compare data files."""
```

### Step 6: Update CLI registry and categories in __init__.py

Update `COMMAND_CATEGORIES`:

```python
COMMAND_CATEGORIES = {
    "Pipeline": ["run", "repro", "status", "verify", "commit"],
    "Sync": ["checkout", "fetch", "get", "pull", "push", "remote", "track"],
    "Inspection": ["dag", "diff", "history", "list", "metrics", "params", "plots", "show"],
    "Other": [
        "init",
        "export",
        "import-dvc",
        "config",
        "completion",
        "schema",
        "check-ignore",
        "doctor",
    ],
}
```

Update `_LAZY_COMMANDS` - remove `data`, add `fetch`, `diff`, `get`, update `pull`:

```python
_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "init": ("pivot.cli.init", "init", "Initialize a new Pivot project."),
    "run": ("pivot.cli.run", "run", "Execute pipeline stages."),
    "repro": ("pivot.cli.repro", "repro", "Reproduce pipeline with full DAG resolution."),
    "dag": ("pivot.cli.dag", "dag_cmd", "Visualize pipeline DAG."),
    "list": ("pivot.cli.list", "list_cmd", "List registered stages."),
    "export": ("pivot.cli.export", "export", "Export pipeline to DVC YAML format."),
    "import-dvc": (
        "pivot.cli.import_dvc",
        "import_dvc",
        "Import DVC pipeline and convert to Pivot format.",
    ),
    "track": ("pivot.cli.track", "track", "Track files/directories for caching."),
    "status": ("pivot.cli.status", "status", "Show pipeline, tracked files, and remote status."),
    "verify": (
        "pivot.cli.verify",
        "verify",
        "Verify pipeline was reproduced and outputs are available.",
    ),
    "checkout": (
        "pivot.cli.checkout",
        "checkout",
        "Restore tracked files and stage outputs from cache.",
    ),
    "metrics": ("pivot.cli.metrics", "metrics", "Display and compare metrics."),
    "plots": ("pivot.cli.plots", "plots", "Display and compare plots."),
    "params": ("pivot.cli.params", "params", "Display and compare parameters."),
    "remote": ("pivot.cli.remote", "remote", "Manage remote storage for cache synchronization."),
    "push": ("pivot.cli.remote", "push", "Push cached outputs to remote storage."),
    "fetch": ("pivot.cli.remote", "fetch", "Fetch cached outputs from remote to local cache."),
    "pull": ("pivot.cli.remote", "pull", "Pull and restore outputs from remote storage."),
    "diff": ("pivot.cli.data", "diff", "Compare data files against git HEAD."),
    "get": ("pivot.cli.data", "get", "Retrieve files from a specific git revision."),
    "completion": ("pivot.cli.completion", "completion_cmd", "Generate shell completion script."),
    "config": ("pivot.cli.config", "config_cmd", "View and modify Pivot configuration."),
    "history": ("pivot.cli.history", "history", "List recent pipeline runs."),
    "show": ("pivot.cli.history", "show_cmd", "Show details of a specific run."),
    "schema": ("pivot.cli.schema", "schema", "Output JSON Schema for pivot.yaml configuration."),
    "commit": ("pivot.cli.commit", "commit_command", "Commit pending locks from --no-commit runs."),
    "check-ignore": (
        "pivot.cli.check_ignore",
        "check_ignore",
        "Check if paths are ignored by .pivotignore.",
    ),
    "doctor": ("pivot.cli.doctor", "doctor", "Check environment and configuration for issues."),
}
```

---

## Task 2: Update All Tests

**Files:**
- Modify: `tests/remote/test_cli_remote.py`
- Modify: `tests/cli/test_cli_data.py`

### Step 1: Rename pull tests to fetch tests in test_cli_remote.py

Update these test functions (rename and change invocations/assertions):

| Old function name | New function name |
|-------------------|-------------------|
| `test_pull_dry_run_with_targets` | `test_fetch_dry_run_with_targets` |
| `test_pull_dry_run_all` | `test_fetch_dry_run_all` |
| `test_pull_success` | `test_fetch_success` |
| `test_pull_with_errors` | `test_fetch_with_errors` |
| `test_pull_with_stages` | `test_fetch_with_stages` |
| `test_pull_exception_shows_click_error` | `test_fetch_exception_shows_click_error` |

For each test:
- Change `runner.invoke(cli.cli, ["pull", ...])` to `runner.invoke(cli.cli, ["fetch", ...])`
- Change assertions from "Pulled from" to "Fetched from"
- Change assertions from "Would pull" to "Would fetch"

Example for `test_fetch_success`:

```python
def test_fetch_success(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Fetch command downloads files and shows summary."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )

        mock_pull = mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=3, skipped=1, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        result = runner.invoke(cli.cli, ["fetch"])

        assert result.exit_code == 0
        assert "Fetched from 'origin': 3 transferred, 1 skipped, 0 failed" in result.output
        mock_pull.assert_called_once()
```

### Step 2: Add tests for new pull command in test_cli_remote.py

```python
# =============================================================================
# Pull Command Tests (fetch + checkout)
# =============================================================================


def test_pull_force_and_only_missing_conflict(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pull errors when both --force and --only-missing specified."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        result = runner.invoke(cli.cli, ["pull", "--force", "--only-missing"])

        assert result.exit_code != 0
        assert "--force and --only-missing are mutually exclusive" in result.output


def test_pull_dry_run_does_not_checkout(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull --dry-run only shows fetch info, doesn't checkout."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        monkeypatch.setattr(project, "_project_root_cache", None)

        mock_remote = mocker.MagicMock()

        async def mock_list_hashes() -> set[str]:
            return {"hash1", "hash2"}

        mock_remote.list_hashes = mock_list_hashes

        mocker.patch.object(config_mod, "get_cache_dir", return_value=tmp_path / ".pivot/cache")
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mock_remote, "origin"),
        )
        mocker.patch.object(transfer, "get_local_cache_hashes", return_value=set())

        result = runner.invoke(cli.cli, ["pull", "--dry-run"])

        assert result.exit_code == 0
        assert "Would fetch" in result.output


def test_pull_success_fetches_and_checks_out(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull fetches from remote then checks out to workspace."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        state_dir = tmp_path / ".pivot"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(config_mod, "get_state_dir", return_value=state_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=2, skipped=0, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        # Mock checkout to verify it's called
        from pivot.cli import checkout as checkout_mod

        mock_checkout = mocker.patch.object(checkout_mod, "checkout", autospec=True)

        result = runner.invoke(cli.cli, ["pull"])

        assert result.exit_code == 0
        assert "Fetched from 'origin'" in result.output
        mock_checkout.assert_called_once()


def test_pull_stops_on_fetch_failure(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull exits without checkout when fetch has failures."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(
                transferred=1, skipped=0, failed=1, errors=["Download failed: hash1"]
            ),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        # Mock checkout to verify it's NOT called
        from pivot.cli import checkout as checkout_mod

        mock_checkout = mocker.patch.object(checkout_mod, "checkout", autospec=True)

        result = runner.invoke(cli.cli, ["pull"])

        assert result.exit_code == 1
        assert "1 failed" in result.output
        mock_checkout.assert_not_called()


def test_pull_passes_targets_to_fetch_and_checkout(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pull passes same targets to both fetch and checkout."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = tmp_path / ".pivot" / "cache"
        state_dir = tmp_path / ".pivot"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(project, "_project_root_cache", None)

        mocker.patch.object(config_mod, "get_cache_dir", return_value=cache_dir)
        mocker.patch.object(config_mod, "get_state_dir", return_value=state_dir)
        mocker.patch.object(
            transfer,
            "create_remote_from_name",
            return_value=(mocker.MagicMock(), "origin"),
        )
        mock_transfer_pull = mocker.patch.object(
            transfer,
            "pull",
            return_value=TransferSummary(transferred=1, skipped=0, failed=0, errors=[]),
        )

        mock_state_db = mocker.MagicMock()
        mocker.patch.object(state, "StateDB", return_value=mock_state_db)

        from pivot.cli import checkout as checkout_mod

        mock_checkout = mocker.patch.object(checkout_mod, "checkout", autospec=True)

        result = runner.invoke(cli.cli, ["pull", "train_model", "evaluate"])

        assert result.exit_code == 0
        # Verify targets passed to transfer.pull
        call_args = mock_transfer_pull.call_args
        assert call_args.args[5] == ["train_model", "evaluate"]
        # Verify targets passed to checkout (via ctx.invoke, check call kwargs)
        checkout_call = mock_checkout.call_args
        assert checkout_call.kwargs.get("targets") == ("train_model", "evaluate")
```

### Step 3: Update data diff/get tests in test_cli_data.py

Update all test invocations from `["data", "diff", ...]` to `["diff", ...]` and `["data", "get", ...]` to `["get", ...]`:

| Test function | Change |
|---------------|--------|
| `test_data_diff_help` → `test_diff_help` | `["data", "diff", "--help"]` → `["diff", "--help"]` |
| `test_data_group_help` | DELETE |
| `test_data_in_main_help` | DELETE |
| `test_data_diff_no_stages` → `test_diff_no_stages` | Update invoke |
| `test_data_diff_key_and_positional_conflict` → `test_diff_key_and_positional_conflict` | Update invoke |
| `test_data_diff_requires_targets` → `test_diff_requires_targets` | Update invoke |
| `test_data_diff_csv_file` → `test_diff_csv_file` | Update invoke |
| `test_data_diff_json_output` → `test_diff_json_output` | Update invoke |
| `test_data_diff_key_columns` → `test_diff_key_columns` | Update invoke |
| `test_data_diff_positional` → `test_diff_positional` | Update invoke |
| `test_data_diff_no_changes_message` → `test_diff_no_changes_message` | Update invoke |
| `test_data_diff_json_empty_returns_valid_json` → `test_diff_json_empty_returns_valid_json` | Update invoke |
| `test_data_get_stage_output` → `test_get_stage_output` | Update invoke |
| `test_data_get_checkout_mode` → `test_get_checkout_mode` | Update invoke |
| `test_data_diff_without_prior_run_discovers_pipeline` → `test_diff_without_prior_run_discovers_pipeline` | Update invoke |
| `test_data_get_without_prior_run_discovers_pipeline` → `test_get_without_prior_run_discovers_pipeline` | Update invoke |

Add tests for new top-level commands appearing in main help:

```python
def test_diff_in_main_help(runner: click.testing.CliRunner) -> None:
    """diff command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "diff" in result.output


def test_get_in_main_help(runner: click.testing.CliRunner) -> None:
    """get command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "get" in result.output
```

---

## Task 3: Final Verification

**Files:** None (verification only)

### Step 1: Run all tests

```bash
uv run pytest tests/ -n auto
```

Expected: All tests pass

### Step 2: Run linting and type checks

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright
```

Expected: No errors

### Step 3: Verify CLI help output

```bash
uv run pivot --help
```

Expected structure:
```
Pipeline Commands:
  commit, repro, run, status, verify

Sync Commands:
  checkout, fetch, get, pull, push, remote, track

Inspection Commands:
  dag, diff, history, list, metrics, params, plots, show

Other Commands:
  check-ignore, completion, config, doctor, export, import-dvc, init, schema
```

### Step 4: Commit

```bash
jj describe -m "feat(cli): redesign pull/fetch commands and reorganize categories

BREAKING CHANGE: pivot pull now fetches AND checks out (like git/dvc pull)

- Rename pull to fetch (downloads to cache only)
- New pull = fetch + checkout (matches git/dvc semantics)
- Promote 'data diff' to top-level 'diff' command
- Promote 'data get' to top-level 'get' command
- Remove 'data' command group
- Reorganize CLI categories: Pipeline, Sync, Inspection, Other

Closes #336"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | All code changes (rename pull→fetch, new pull, promote diff/get, update registry) |
| 2 | All test updates (rename tests, add new pull tests, update diff/get tests) |
| 3 | Final verification (tests, linting, types, CLI help, commit) |
