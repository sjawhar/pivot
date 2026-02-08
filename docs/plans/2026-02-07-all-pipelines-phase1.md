# `--all` Flag Phase 1: `verify --all` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--all` flag to `pivot verify` that discovers all pipelines in the project, merges them into one unified DAG, and verifies all stages — with correct per-stage `state_dir` routing for lock files and StateDB access.

**Architecture:** Extend `discover_pipeline()` with `all_pipelines=True` that globs all pipeline configs, loads each, and `include()`s them into a synthetic root Pipeline. Gate `--all` in the `pivot_command` decorator via `allow_all=True`. Replace `config.get_state_dir()` in verify/status code paths with per-stage `state_dir` from `RegistryStageInfo`.

**Tech Stack:** Python 3.13+, Click, pytest

**Design doc:** `docs/plans/2026-02-06-all-pipelines-flag-design.md`

---

### Task 1: Make `_glob_all_pipelines` public and deduplicate by directory

The existing `_glob_all_pipelines` in `pipeline.py` is private and has a bug: it can return both `pipeline.py` AND `pivot.yaml` from the same directory (the bug-finder flagged this). We need to make it public, move it to `discovery.py`, and deduplicate.

**Files:**
- Modify: `src/pivot/discovery.py`
- Modify: `src/pivot/pipeline/pipeline.py`
- Test: `tests/core/test_discovery.py`

**Step 1: Write failing tests for `glob_all_pipelines`**

Add to `tests/core/test_discovery.py`:

```python
# =============================================================================
# glob_all_pipelines Tests
# =============================================================================


def test_glob_all_pipelines_finds_pipeline_py(set_project_root: pathlib.Path) -> None:
    """glob_all_pipelines discovers pipeline.py files in subdirectories."""
    sub = set_project_root / "sub"
    sub.mkdir()
    (sub / "pipeline.py").write_text("pipeline = None")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 1
    assert result[0] == sub / "pipeline.py"


def test_glob_all_pipelines_finds_pivot_yaml(set_project_root: pathlib.Path) -> None:
    """glob_all_pipelines discovers pivot.yaml files in subdirectories."""
    sub = set_project_root / "sub"
    sub.mkdir()
    (sub / "pivot.yaml").write_text("stages: {}")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 1
    assert result[0] == sub / "pivot.yaml"


def test_glob_all_pipelines_skips_excluded_dirs(set_project_root: pathlib.Path) -> None:
    """glob_all_pipelines skips .venv, __pycache__, .git, etc."""
    venv = set_project_root / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "pipeline.py").write_text("pipeline = None")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 0


def test_glob_all_pipelines_deduplicates_by_directory(
    set_project_root: pathlib.Path,
) -> None:
    """glob_all_pipelines raises DiscoveryError if directory has both config types.

    A directory with both pipeline.py and pivot.yaml is ambiguous. This is the
    same constraint as find_config_in_dir but applied during project-wide scan.
    """
    sub = set_project_root / "sub"
    sub.mkdir()
    (sub / "pipeline.py").write_text("pipeline = None")
    (sub / "pivot.yaml").write_text("stages: {}")

    with pytest.raises(discovery.DiscoveryError, match="both"):
        discovery.glob_all_pipelines(set_project_root)


def test_glob_all_pipelines_finds_multiple_pipelines(
    set_project_root: pathlib.Path,
) -> None:
    """glob_all_pipelines finds pipelines in multiple subdirectories."""
    for name in ("alpha", "beta", "gamma"):
        sub = set_project_root / name
        sub.mkdir()
        (sub / "pipeline.py").write_text("pipeline = None")

    result = discovery.glob_all_pipelines(set_project_root)

    assert len(result) == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_discovery.py -k "glob_all_pipelines" -v`
Expected: FAIL — `discovery.glob_all_pipelines` does not exist

**Step 3: Implement `glob_all_pipelines` in discovery.py**

Add to `src/pivot/discovery.py` after `load_pipeline_from_path`:

```python
# Directories excluded from all-pipelines scan
_SCAN_EXCLUDE_DIRS = frozenset(
    {
        ".pivot",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".ruff_cache",
    }
)


def glob_all_pipelines(project_root: pathlib.Path) -> list[pathlib.Path]:
    """Find all pipeline config files in the project.

    Scans project_root recursively for pipeline.py and pivot.yaml/yml files,
    skipping common non-project directories (.venv, __pycache__, etc.).

    Deduplicates by directory: if a directory contains both pipeline.py and
    pivot.yaml, raises DiscoveryError (same constraint as find_config_in_dir).

    Args:
        project_root: Project root directory to scan.

    Returns:
        List of paths to pipeline config files.

    Raises:
        DiscoveryError: If any directory has both pipeline.py and pivot.yaml.
    """
    # Collect all candidate paths grouped by directory
    by_dir: dict[pathlib.Path, list[pathlib.Path]] = {}
    target_names = (PIPELINE_PY_NAME, *PIVOT_YAML_NAMES)
    for name in target_names:
        for path in project_root.rglob(name):
            if any(part in _SCAN_EXCLUDE_DIRS for part in path.parts):
                continue
            by_dir.setdefault(path.parent, []).append(path)

    # Validate: no directory should have both config types
    results = list[pathlib.Path]()
    for directory, paths in by_dir.items():
        if len(paths) > 1:
            # Use find_config_in_dir which raises DiscoveryError for ambiguity
            find_config_in_dir(directory)
        results.append(paths[0])

    return results
```

**Step 4: Update `_glob_all_pipelines` in pipeline.py to call the public version**

In `src/pivot/pipeline/pipeline.py`, replace the private `_glob_all_pipelines` function (lines 148-156) with a delegation to the new public function:

```python
def _glob_all_pipelines(project_root: pathlib.Path) -> list[pathlib.Path]:
    """Glob all pipeline.py and pivot.yaml/yml files in the project."""
    return discovery.glob_all_pipelines(project_root)
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_discovery.py -k "glob_all_pipelines" -v`
Expected: PASS

**Step 6: Run full test suite to check for regressions**

Run: `uv run pytest tests/core/test_discovery.py tests/pipeline/test_pipeline.py -v`
Expected: All pass

---

### Task 2: Extend `discover_pipeline` with `all_pipelines` parameter

**Files:**
- Modify: `src/pivot/discovery.py:55-119` (`discover_pipeline` function)
- Test: `tests/core/test_discovery.py`

**Step 1: Write failing tests for `discover_pipeline(all_pipelines=True)`**

Add to `tests/core/test_discovery.py`:

```python
# =============================================================================
# discover_pipeline(all_pipelines=True) Tests
# =============================================================================


def test_discover_all_pipelines_combines_stages(
    set_project_root: pathlib.Path,
) -> None:
    """discover_pipeline(all_pipelines=True) creates a combined pipeline with all stages."""
    from pivot.pipeline.pipeline import Pipeline

    # Create two sub-pipelines with distinct stages
    for name, stage_name in [("alpha", "stage_a"), ("beta", "stage_b")]:
        sub = set_project_root / name
        sub.mkdir()
        code = f"""\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("{name}", root=__import__("pathlib").Path(__file__).parent)

def _stage():
    pass

pipeline.register(_stage, name="{stage_name}")
"""
        (sub / "pipeline.py").write_text(code)

    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is not None
    assert isinstance(result, Pipeline)
    assert "stage_a" in result.list_stages()
    assert "stage_b" in result.list_stages()


def test_discover_all_pipelines_preserves_state_dir(
    set_project_root: pathlib.Path,
) -> None:
    """Each included stage retains its original pipeline's state_dir."""
    alpha_dir = set_project_root / "alpha"
    alpha_dir.mkdir()
    code_a = """\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("alpha", root=__import__("pathlib").Path(__file__).parent)

def _stage():
    pass

pipeline.register(_stage, name="stage_a")
"""
    (alpha_dir / "pipeline.py").write_text(code_a)

    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is not None
    info = result.get("stage_a")
    assert info["state_dir"] == alpha_dir / ".pivot"


def test_discover_all_pipelines_name_collision_auto_prefixes(
    set_project_root: pathlib.Path,
) -> None:
    """Name collisions across pipelines are resolved by auto-prefixing with pipeline name.

    Note: The original plan specified PipelineConfigError on collision, but the
    implementation uses auto-prefix instead — this avoids errors when pipelines
    share common stage names like 'train'.
    """
    for name in ("alpha", "beta"):
        sub = set_project_root / name
        sub.mkdir()
        code = f"""\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("{name}", root=__import__("pathlib").Path(__file__).parent)

def _stage():
    pass

pipeline.register(_stage, name="duplicate_name")
"""
        (sub / "pipeline.py").write_text(code)

    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)
    assert result is not None
    # One pipeline gets bare name, the other gets prefixed
    assert "duplicate_name" in result.list_stages()


def test_discover_all_pipelines_returns_none_when_empty(
    set_project_root: pathlib.Path,
) -> None:
    """discover_pipeline(all_pipelines=True) returns None when no pipelines found."""
    result = discovery.discover_pipeline(set_project_root, all_pipelines=True)

    assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_discovery.py -k "discover_all_pipelines" -v`
Expected: FAIL — `discover_pipeline` does not accept `all_pipelines`

**Step 3: Implement the `all_pipelines` parameter**

Modify `discover_pipeline` in `src/pivot/discovery.py:55`:

```python
def discover_pipeline(
    project_root: pathlib.Path | None = None,
    *,
    all_pipelines: bool = False,
) -> Pipeline | None:
    """Discover and return Pipeline from pivot.yaml or pipeline.py.

    Looks for pipeline config in this order:
    1. Current working directory (if within project root)
    2. Project root

    In each location, checks for:
    - pivot.yaml (or pivot.yml) - creates implicit Pipeline
    - pipeline.py - looks for `pipeline` variable (Pipeline instance)

    When all_pipelines=True, discovers ALL pipeline config files in the project,
    loads each, and merges them into a single Pipeline via include(). The combined
    pipeline contains stages from all discovered pipelines, each retaining its
    original state_dir.

    Args:
        project_root: Override project root (default: auto-detect)
        all_pipelines: If True, discover and combine all pipelines in project.

    Returns:
        Pipeline instance, or None if nothing found

    Raises:
        DiscoveryError: If discovery fails, or if both config types exist
    """
    _t = metrics.start()
    try:
        root = project_root or project.get_project_root()

        if all_pipelines:
            return _discover_all_pipelines(root)

        # ... existing single-pipeline discovery logic unchanged ...
```

Add the new private function:

```python
def _discover_all_pipelines(root: pathlib.Path) -> Pipeline | None:
    """Discover all pipelines and combine into one.

    Globs all pipeline config files, loads each, and merges via include().
    """
    from pivot.pipeline.pipeline import Pipeline

    config_paths = glob_all_pipelines(root)
    if not config_paths:
        return None

    pipelines = list[Pipeline]()
    for path in config_paths:
        pipeline = load_pipeline_from_path(path)
        if pipeline is not None:
            pipelines.append(pipeline)

    if not pipelines:
        return None

    combined = Pipeline("all", root=root)
    for pipeline in pipelines:
        combined.include(pipeline)  # Auto-prefixes on name collision

    logger.info(
        f"Discovered {len(pipelines)} pipelines with {len(combined.list_stages())} total stages"
    )
    return combined
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_discovery.py -k "discover_all_pipelines" -v`
Expected: PASS

**Step 5: Run full discovery test suite**

Run: `uv run pytest tests/core/test_discovery.py -v`
Expected: All pass (existing tests unchanged since default `all_pipelines=False`)

---

### Task 3: Add `--all` flag to `pivot_command` decorator (gated with `allow_all`)

**Files:**
- Modify: `src/pivot/cli/decorators.py:55-103`
- Test: `tests/cli/test_cli_decorators.py` (or whatever file tests decorators — check first)

**Step 1: Write failing test**

Check which file tests the decorator, then add:

```python
def test_pivot_command_all_flag_when_allowed(
    runner: CliRunner, set_project_root: pathlib.Path
) -> None:
    """Commands with allow_all=True accept --all flag."""
    from pivot.cli import decorators as cli_decorators

    @cli_decorators.pivot_command(auto_discover=False, allow_all=True)
    @click.pass_context
    def test_cmd(ctx: click.Context) -> None:
        click.echo("ok")

    result = runner.invoke(test_cmd, ["--all"])
    assert result.exit_code == 0
    assert "ok" in result.output


def test_pivot_command_all_flag_when_not_allowed(runner: CliRunner) -> None:
    """Commands without allow_all=True do not accept --all flag."""
    from pivot.cli import decorators as cli_decorators

    @cli_decorators.pivot_command(auto_discover=False)
    def test_cmd() -> None:
        click.echo("ok")

    result = runner.invoke(test_cmd, ["--all"])
    assert result.exit_code != 0
    assert "no such option" in result.output.lower() or "Error" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/ -k "pivot_command_all_flag" -v`
Expected: FAIL

**Step 3: Implement the `allow_all` parameter**

Modify `pivot_command` in `src/pivot/cli/decorators.py:55`:

```python
def pivot_command(
    name: str | None = None,
    *,
    auto_discover: bool = True,
    allow_all: bool = False,
    **attrs: Any,
) -> Callable[[Callable[..., Any]], click.Command]:
    """Create a Click command with Pivot error handling and optional auto-discovery.

    Args:
        name: Optional command name (defaults to function name)
        auto_discover: If True (default), automatically discover and register
            stages before running the command.
        allow_all: If True, add --all flag for multi-pipeline discovery.
        **attrs: Additional arguments passed to click.command()
    """

    def decorator(func: Callable[..., Any]) -> click.Command:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _t_total = metrics.start()
            try:
                # Check if Pipeline is already in context (e.g., when invoking subcommand)
                if auto_discover and not _has_pipeline_in_context():
                    try:
                        _t_discover = metrics.start()
                        try:
                            # Check for --all flag in Click context
                            ctx = click.get_current_context(silent=True)
                            use_all = (
                                allow_all
                                and ctx is not None
                                and ctx.params.get("all_pipelines", False)
                            )
                            pipeline = discovery.discover_pipeline(
                                all_pipelines=use_all,
                            )
                            if pipeline is not None:
                                store_pipeline_in_context(pipeline)
                        finally:
                            metrics.end("cli.discover", _t_discover)
                    except discovery.DiscoveryError as e:
                        raise click.ClickException(str(e)) from e
                return func(*args, **kwargs)
            finally:
                metrics.end("cli.total", _t_total)
                if os.environ.get("PIVOT_METRICS"):
                    _print_metrics_summary()

        wrapped = with_error_handling(wrapper)
        cmd = click.command(name=name, **attrs)(wrapped)

        # Add --all option if allowed
        if allow_all:
            cmd = click.option(
                "--all",
                "all_pipelines",
                is_flag=True,
                default=False,
                help="Run across all pipelines in the project.",
            )(cmd)

        return cmd

    return decorator
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/ -k "pivot_command_all_flag" -v`
Expected: PASS

---

### Task 4: Per-stage `state_dir` in `status.py`

The `verify` command calls `status.get_pipeline_status()` and `status.get_pipeline_explanations()` which both use a global `config.get_state_dir()`. Fix them to use per-stage `state_dir` from `all_stages`.

**Files:**
- Modify: `src/pivot/status.py:197,295`
- Test: `tests/status/test_status.py` (check existing tests first)

**Step 1: Write failing test**

```python
def test_get_pipeline_status_uses_per_stage_state_dir(
    set_project_root: pathlib.Path,
) -> None:
    """get_pipeline_status uses each stage's state_dir, not the global one.

    When stages from different pipelines have different state_dirs,
    the status check must read lock files from the correct location.
    """
    # This test verifies the state_dir is passed from stage_info, not config.
    # Create a stage whose state_dir differs from config.get_state_dir().
    # The lock file is in the stage's state_dir — if status reads from the
    # global state_dir, it won't find the lock and will report stale.
    ...
```

Note: The exact test setup depends on existing test patterns in `tests/status/`. Read those first and follow the same fixture patterns.

**Step 2: Implement the fix**

In `src/pivot/status.py`, change both `get_pipeline_explanations` (line 197) and `get_pipeline_status` (line 295) to pass per-stage `state_dir`:

```python
# In _get_explanations_in_parallel, line 129 changes from:
#     state_dir,
# to:
#     stage_info["state_dir"],
```

The key change: in `_get_explanations_in_parallel` (line 102), instead of passing the same `state_dir` to every `pool.submit(explain.get_stage_explanation, ...)` call, pass `stage_info["state_dir"]`:

```python
def _get_explanations_in_parallel(
    execution_order: list[str],
    overrides: parameters.ParamsOverrides | None,
    all_stages: dict[str, RegistryStageInfo],
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
) -> dict[str, StageExplanation]:
```

Remove the `state_dir` parameter from `_get_explanations_in_parallel` entirely — each stage's `state_dir` comes from `all_stages[stage_name]["state_dir"]`.

Update both callers (`get_pipeline_explanations` line 200 and `get_pipeline_status` line 298) to stop passing `state_dir`.

**Step 3: Update `get_pipeline_explanations` and `get_pipeline_status`**

Remove the `state_dir = config.get_state_dir()` line from both functions (lines 197 and 295). The `state_dir` is no longer needed at that level — it's read per-stage inside `_get_explanations_in_parallel`.

**Step 4: Run tests**

Run: `uv run pytest tests/status/ -v`
Expected: All pass

---

### Task 5: Per-stage `state_dir` in `cli/verify.py`

**Files:**
- Modify: `src/pivot/cli/verify.py:65-101,159-164,308`
- Test: `tests/cli/test_verify.py`

**Step 1: Write failing integration test**

Add to `tests/cli/test_verify.py`:

```python
def test_verify_all_flag_multi_pipeline(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """verify --all discovers all pipelines and checks each one's lock files.

    Sets up two pipelines with different state_dirs, creates lock files in each,
    and verifies that --all finds and checks both.
    """
    with isolated_pivot_dir(runner, tmp_path):
        # Create two sub-pipelines
        for name, stage_name in [("alpha", "stage_a"), ("beta", "stage_b")]:
            sub = tmp_path / name
            sub.mkdir()
            (sub / ".pivot").mkdir()
            # ... create pipeline.py, register stage, create lock file ...

        result = runner.invoke(cli.cli, ["verify", "--all"])

        assert "stage_a" in result.output
        assert "stage_b" in result.output
```

Note: The exact test setup depends heavily on existing `test_verify.py` patterns. Read them first and replicate the lock file creation pattern from `make_valid_lock_content`.

**Step 2: Update `verify` command to accept `--all`**

In `src/pivot/cli/verify.py:270`, change:

```python
@cli_decorators.pivot_command()
```

to:

```python
@cli_decorators.pivot_command(allow_all=True)
```

**Step 3: Update `_get_stage_lock_hashes` to use per-stage `state_dir`**

In `src/pivot/cli/verify.py:65-67`, change `_get_stage_lock_hashes` to get `state_dir` from the stage's registry info instead of a parameter:

```python
def _get_stage_lock_hashes(
    stage_name: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Get output and dep file hashes from a stage's lock file."""
    stage_info = cli_helpers.get_stage(stage_name)
    state_dir = stage_info["state_dir"]
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    ...
```

Update `_get_stage_missing_hashes` (line 104) similarly — remove the `state_dir` parameter, get it from the registry.

Update `_verify_stages` (line 159) — remove the `state_dir` parameter.

Update the call in `verify()` (line 324) — remove `state_dir` argument.

Remove the `state_dir = config.get_state_dir()` line at line 308.

**Step 4: Run tests**

Run: `uv run pytest tests/cli/test_verify.py -v`
Expected: All pass

---

### Task 6: Quality checks and final verification

**Files:** None (validation only)

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All pass, no regressions

**Step 2: Run type checker**

Run: `uv run basedpyright`
Expected: No new errors

**Step 3: Run linter and formatter**

Run: `uv run ruff format . && uv run ruff check .`
Expected: Clean

**Step 4: Manual smoke test on eval-pipeline**

```bash
cd ~/eval-pipeline/pivot
uv run pivot verify --all
```

Expected: Discovers all 6 pipelines, verifies stages from each with correct state_dir routing.

---

## Summary of changes by file

| File | Change |
|------|--------|
| `src/pivot/discovery.py` | Add `glob_all_pipelines()`, `_discover_all_pipelines()`, extend `discover_pipeline()` signature |
| `src/pivot/cli/decorators.py` | Add `allow_all` param, `--all` Click option, pass `all_pipelines` to discovery |
| `src/pivot/cli/verify.py` | Add `allow_all=True`, remove global `state_dir`, use per-stage lookup |
| `src/pivot/status.py` | Remove `state_dir` param from `_get_explanations_in_parallel`, use per-stage |
| `src/pivot/pipeline/pipeline.py` | Delegate `_glob_all_pipelines` to `discovery.glob_all_pipelines` |
| `tests/core/test_discovery.py` | Tests for `glob_all_pipelines`, `discover_pipeline(all_pipelines=True)` |
| `tests/cli/test_verify.py` | Integration test for `verify --all` with mixed state_dirs |
