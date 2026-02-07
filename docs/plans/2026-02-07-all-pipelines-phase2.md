# `--all` Flag Phase 2: Enable Remaining Commands

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable `--all` on `repro`, `status`, `commit`, `push`, `pull` and fix all remaining per-stage `state_dir` call sites so the coordinator correctly routes lock files and StateDB writes for multi-pipeline projects.

**Architecture:** Apply the same `allow_all=True` gating from Phase 1 to the remaining commands. Replace every `config.get_state_dir()` in engine, executor, checkout, and remote code with per-stage `state_dir` from `RegistryStageInfo`. Store `all_pipelines` flag in the engine for watch mode reload. Add pipeline config files to watch paths.

**Tech Stack:** Python 3.13+, Click, anyio, loky, pytest

**Depends on:** Phase 1 plan (`docs/plans/2026-02-07-all-pipelines-phase1.md`) must be complete.

**Design doc:** `docs/plans/2026-02-06-all-pipelines-flag-design.md`

---

### Task 1: Enable `--all` on `repro`, `status`, `commit`

These commands already use `@cli_decorators.pivot_command()` with `auto_discover=True`, so adding `allow_all=True` is a one-line change per command.

**Files:**
- Modify: `src/pivot/cli/repro.py:741`
- Modify: `src/pivot/cli/status.py:29`
- Modify: `src/pivot/cli/commit.py:12`
- Test: `tests/cli/test_repro.py`, `tests/cli/test_cli_status.py`, `tests/cli/test_cli_commit.py`

**Step 1: Write failing tests**

Add to each CLI test file a test that `--all` is accepted:

```python
# In tests/cli/test_repro.py
def test_repro_accepts_all_flag(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --all is accepted and triggers all-pipeline discovery."""
    with isolated_pivot_dir(runner, tmp_path):
        # Create two sub-pipelines
        for name, stage_name in [("alpha", "stage_a"), ("beta", "stage_b")]:
            sub = tmp_path / name
            sub.mkdir()
            (sub / ".pivot").mkdir()
            create_pipeline_py(
                [_helper_noop_stage],
                path=sub,
                names={"_helper_noop_stage": stage_name},
            )

        result = runner.invoke(cli.cli, ["repro", "--all", "--dry-run"])
        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" in result.output
```

Similar tests for `status --all` and `commit --all`.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/test_repro.py -k "all_flag" -v`
Expected: FAIL — `--all` not recognized

**Step 3: Add `allow_all=True` to each command**

In `src/pivot/cli/repro.py:741`:
```python
@cli_decorators.pivot_command(allow_all=True)
```

In `src/pivot/cli/status.py:29`:
```python
@cli_decorators.pivot_command(allow_all=True)
```

In `src/pivot/cli/commit.py:12`:
```python
@cli_decorators.pivot_command("commit", allow_all=True)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_repro.py tests/cli/test_cli_status.py tests/cli/test_cli_commit.py -k "all_flag" -v`
Expected: PASS

---

### Task 2: Engine per-stage StateDB — deferred writes

The engine opens a single StateDB at `config.get_state_db_path()` (line 557) and applies all deferred writes there. When stages from different pipelines have different `state_dir` values, writes go to the wrong database.

**The worker already uses per-stage `state_dir`** (via `prepare_worker_info` line 333). The coordinator just needs to route deferred writes correctly.

**Files:**
- Modify: `src/pivot/engine/engine.py:533,557,567,607-613`
- Test: `tests/engine/test_engine.py`

**Step 1: Write failing test**

```python
def test_deferred_writes_go_to_correct_state_db(
    set_project_root: pathlib.Path,
) -> None:
    """Deferred writes for stages with different state_dirs go to the right DB.

    When stage_a has state_dir=alpha/.pivot and stage_b has state_dir=beta/.pivot,
    deferred writes from stage_a must go to alpha/.pivot/state.db, not the
    project-level state.db.
    """
    ...
```

Note: The exact test depends on existing engine test patterns. The key assertion is that after applying deferred writes, the generation bump appears in the stage's own StateDB, not the project-level one.

**Step 2: Implement per-stage StateDB routing**

In `src/pivot/engine/engine.py`, the `_orchestrate_execution` method (line 460+):

Replace the single StateDB with a cache of StateDB connections keyed by `state_dir`:

```python
# Line 533: remove global state_dir assignment
# state_dir = config.get_state_dir()  # DELETE this line
# Keep it only for default_state_dir fallback:
default_state_dir = config.get_state_dir()

# Line 557: remove single state_db_path
# state_db_path = config.get_state_db_path()  # DELETE this line
```

Replace the single `with state_mod.StateDB(state_db_path) as state_db:` block (line 567) with a StateDB connection cache:

```python
# Instead of opening one StateDB, manage a cache
state_dbs = dict[pathlib.Path, state_mod.StateDB]()

def _get_state_db(stage_state_dir: pathlib.Path) -> state_mod.StateDB:
    """Get or open a StateDB for the given state_dir."""
    if stage_state_dir not in state_dbs:
        db_path = stage_state_dir / "state.db"
        state_dbs[stage_state_dir] = state_mod.StateDB(db_path)
        state_dbs[stage_state_dir].open()
    return state_dbs[stage_state_dir]
```

At the deferred writes application (lines 607-613), look up the stage's `state_dir`:

```python
if result["status"] == StageStatus.RAN and not no_commit:
    stage_info = self._get_stage(stage_name)
    output_paths = [str(out.path) for out in stage_info["outs"]]
    stage_state_dir = stage_info["state_dir"] or default_state_dir
    stage_db = _get_state_db(stage_state_dir)
    executor_core.apply_deferred_writes(
        stage_name, output_paths, result, stage_db
    )
```

Ensure all StateDBs are closed in the `finally` block:

```python
finally:
    for db in state_dbs.values():
        db.close()
```

Update `_start_ready_stages` call (line 580) to pass `default_state_dir` instead of `state_dir`:

```python
state_dir=default_state_dir,
```

Ensure `default_state_dir.mkdir(parents=True, exist_ok=True)` still happens (line 537).

**Step 3: Run tests**

Run: `uv run pytest tests/engine/ -v`
Expected: All pass

---

### Task 3: Engine per-stage StateDB — run history

The `_write_run_history` method (line 1066) reads lock files from the global `state_dir` and writes the run manifest to a single StateDB. Fix both.

**Files:**
- Modify: `src/pivot/engine/engine.py:1066-1113`
- Test: `tests/engine/test_engine.py`

**Step 1: Implement the fix**

In `_write_run_history` (line 1081), replace:

```python
state_dir = config.get_state_dir()
```

with per-stage lookup:

```python
# Read lock files from each stage's own state_dir
for name, summary in results.items():
    stage_info = self._get_stage(name)
    stage_state_dir = stage_info["state_dir"] or config.get_state_dir()
    stage_lock = lock.StageLock(name, lock.get_stages_dir(stage_state_dir))
    ...
```

For the run manifest write (line 1111), write to the project-level StateDB since run history is project-scoped:

```python
with state_mod.StateDB(config.get_state_db_path()) as state_db:
    state_db.write_run(manifest)
    state_db.prune_runs(retention)
```

This line stays unchanged — run history is intentionally project-level.

**Step 2: Run tests**

Run: `uv run pytest tests/engine/ -v`
Expected: All pass

---

### Task 4: `executor/core.py` — per-stage `state_dir` for incremental output check

`check_uncached_incremental_outputs` (line 448) uses `config.get_state_dir()` for all stages. It already has `all_stages` dict with per-stage `state_dir`.

**Files:**
- Modify: `src/pivot/executor/core.py:448-477`
- Test: `tests/execution/test_executor_core.py` (check existing file name)

**Step 1: Implement the fix**

Replace line 457:

```python
state_dir = config.get_state_dir()
```

with per-stage lookup inside the loop (line 460):

```python
for stage_name in execution_order:
    stage_info = all_stages[stage_name]
    stage_state_dir = stage_info["state_dir"] or config.get_state_dir()
    stage_outs = stage_info["outs"]

    # Read lock file from stage's state_dir
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_state_dir))
    ...
```

**Step 2: Run tests**

Run: `uv run pytest tests/execution/ -v`
Expected: All pass

---

### Task 5: `executor/commit.py` — per-stage `state_dir` for pending lock commit

`commit_pending` (line 14) writes production locks to a single `state_dir` and opens one StateDB. When stages from different pipelines have different state dirs, locks go to the wrong place.

**Files:**
- Modify: `src/pivot/executor/commit.py:14-61`
- Test: `tests/execution/test_executor_commit.py` (check existing)

**Step 1: Understand the current flow**

`commit_pending` reads pending locks from `project_root / ".pivot" / "pending" / "stages"` (project-level), then writes production locks to `state_dir / "stages"`. The pending lock location is already project-level and shared. The production lock needs to go to the stage's own `state_dir`.

The challenge: pending locks don't carry `state_dir` info. We need to look up the stage in the registry to find its `state_dir`.

**Step 2: Implement the fix**

Modify `commit_pending` to accept an optional `all_stages` dict for per-stage `state_dir` lookup. When `all_stages` is provided, use `all_stages[stage_name]["state_dir"]` for the production lock path. When not provided (backward compat), fall back to `config.get_state_dir()`.

```python
def commit_pending(
    all_stages: dict[str, RegistryStageInfo] | None = None,
) -> list[str]:
    """Promote pending locks to production and update StateDB."""
    project_root = project.get_project_root()
    pending_stages = lock.list_pending_stages(project_root)
    if not pending_stages:
        return []

    default_state_dir = config.get_state_dir()
    committed = list[str]()

    # Group stages by state_dir for efficient StateDB access
    stages_by_state_dir = dict[pathlib.Path, list[str]]()
    for stage_name in pending_stages:
        if all_stages and stage_name in all_stages:
            state_dir = all_stages[stage_name]["state_dir"] or default_state_dir
        else:
            state_dir = default_state_dir
        stages_by_state_dir.setdefault(state_dir, []).append(stage_name)

    for state_dir, stage_names in stages_by_state_dir.items():
        with state_mod.StateDB(state_dir / "state.db") as state_db:
            for stage_name in stage_names:
                pending_lock = lock.get_pending_lock(stage_name, project_root)
                pending_data = pending_lock.read()
                if pending_data is None:
                    continue

                production_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
                production_lock.write(pending_data)

                # ... rest of logic unchanged ...
                committed.append(stage_name)

    return committed
```

Update the caller in `src/pivot/cli/commit.py` to pass `all_stages` when the pipeline is available:

```python
# In commit_command, after the pending_state_lock:
try:
    all_stages = cli_helpers.get_all_stages()
except Exception:
    all_stages = None  # Graceful fallback if pipeline not loaded
committed = commit.commit_pending(all_stages=all_stages)
```

**Step 3: Run tests**

Run: `uv run pytest tests/execution/ tests/cli/test_cli_commit.py -v`
Expected: All pass

---

### Task 6: `cli/checkout.py` — per-stage `state_dir` for output restoration

`checkout` (line 336) uses `config.get_state_dir()` to find lock files. Fix `_get_stage_output_info` to use per-stage `state_dir`.

**Files:**
- Modify: `src/pivot/cli/checkout.py:35-64,336`
- Test: `tests/cli/test_cli_checkout.py`

**Step 1: Implement the fix**

In `_get_stage_output_info` (line 35), replace the `state_dir` parameter with per-stage lookup:

```python
def _get_stage_output_info() -> dict[str, OutputHash]:
    """Get output hash info from lock files for cached stage outputs only."""
    result = dict[str, OutputHash]()

    for stage_name in cli_helpers.list_stages():
        stage_info = cli_helpers.get_stage(stage_name)
        state_dir = stage_info["state_dir"]
        cached_paths = {
            ...  # unchanged
        }

        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        ...  # rest unchanged
```

Remove `state_dir = config.get_state_dir()` at line 336 and update the call at line 347:

```python
stage_outputs = _get_stage_output_info()  # no state_dir param
```

**Step 2: Run tests**

Run: `uv run pytest tests/cli/test_cli_checkout.py -v`
Expected: All pass

---

### Task 7: `cli/remote.py` — enable `--all` on `push` and `pull`

`push` and `pull` use `auto_discover=False` because they work from lock files. With `--all`, they need the registry to resolve per-stage `state_dir`. Switch to `auto_discover=True` (which is fine even without `--all` — it just becomes a no-op if no pipeline is found) or gate auto-discovery on the `--all` flag.

**Files:**
- Modify: `src/pivot/cli/remote.py:43,112,183`
- Modify: `src/pivot/remote/sync.py:132-185`
- Test: `tests/remote/test_remote_cli.py` (check existing)

**Step 1: Enable `--all` on push, pull, fetch**

Change the decorators:

```python
# push (line 43):
@cli_decorators.pivot_command(allow_all=True)

# fetch (line 112):
@cli_decorators.pivot_command(allow_all=True)

# pull (line 183):
@cli_decorators.pivot_command(allow_all=True)
```

Note: Setting `allow_all=True` without changing `auto_discover` from `False` means we need to handle discovery in the decorator. Looking at the Phase 1 decorator implementation: the `auto_discover` check gates discovery. For push/pull, we want discovery to happen only when `--all` is passed. Update the decorator logic:

```python
# In the decorator wrapper:
if (auto_discover or use_all) and not _has_pipeline_in_context():
```

This way, push/pull with `--all` triggers discovery, but without `--all` they skip it (existing behavior).

**Step 2: Update `get_target_hashes` for per-stage `state_dir`**

In `src/pivot/remote/sync.py:132`, when the registry is available (i.e., `--all` mode), use per-stage `state_dir` for lock file lookup:

```python
def get_target_hashes(
    targets: list[str],
    state_dir: pathlib.Path,
    include_deps: bool = False,
    all_stages: dict[str, RegistryStageInfo] | None = None,
) -> set[str]:
    """Resolve targets to cache hashes.

    When all_stages is provided, uses per-stage state_dir for lock file lookup.
    Otherwise falls back to the provided state_dir.
    """
    ...
    for target in targets:
        if "/" not in target and "\\" not in target:
            # Determine state_dir for this stage
            if all_stages and target in all_stages:
                target_state_dir = all_stages[target]["state_dir"] or state_dir
            else:
                target_state_dir = state_dir
            stage_lock = lock.StageLock(target, lock.get_stages_dir(target_state_dir))
            ...
```

Update callers in `cli/remote.py` to pass `all_stages` when available:

```python
# In push command:
try:
    all_stages = cli_helpers.get_all_stages()
except Exception:
    all_stages = None
local_hashes = transfer.get_target_hashes(
    targets_list, state_dir, include_deps=False, all_stages=all_stages
)
```

**Step 3: Run tests**

Run: `uv run pytest tests/remote/ tests/cli/ -k "push or pull or fetch" -v`
Expected: All pass

---

### Task 8: Watch mode — persist `all_pipelines` flag and reload correctly

The engine's `_reload_registry` (line 1301) calls `discover_pipeline(root)` which only finds one pipeline. In `--all` mode, it needs to re-discover all pipelines.

**Files:**
- Modify: `src/pivot/engine/engine.py:114-116,1301-1332,1162-1192`
- Test: `tests/engine/test_engine.py`

**Step 1: Add `all_pipelines` flag to Engine**

In `Engine.__init__` (line 114), add:

```python
def __init__(self, *, pipeline: Pipeline | None = None, all_pipelines: bool = False) -> None:
    self._pipeline = pipeline
    self._all_pipelines = all_pipelines
    ...
```

**Step 2: Update `_reload_registry` to use the flag**

In `_reload_registry` (line 1321), change:

```python
new_pipeline = discovery.discover_pipeline(root)
```

to:

```python
new_pipeline = discovery.discover_pipeline(root, all_pipelines=self._all_pipelines)
```

**Step 3: Add pipeline config files to watch paths**

In `_handle_code_or_config_changed` (line 1186-1192), after updating watch paths from the graph, also add pipeline config files when in `--all` mode:

```python
# After existing watch path update:
if self._all_pipelines:
    from pivot import discovery
    config_paths = discovery.glob_all_pipelines(project.get_project_root())
    watch_paths.extend(p.parent for p in config_paths)
```

This ensures that adding/removing a `pipeline.py` or `pivot.yaml` triggers a reload.

**Step 4: Pass `all_pipelines` from CLI to Engine**

In `src/pivot/cli/repro.py`, where the Engine is created, pass the flag. Find where `engine.Engine(pipeline=...)` is called and add `all_pipelines`:

```python
# Need to read the --all flag from context and pass to Engine
eng = engine.Engine(pipeline=pipeline, all_pipelines=use_all_pipelines)
```

Check how the repro CLI creates the engine and threads the flag through.

**Step 5: Run tests**

Run: `uv run pytest tests/engine/ -v`
Expected: All pass

---

### Task 9: Integration test — `repro --all` with mixed state_dirs

End-to-end test that exercises the full pipeline: discovery, DAG building, execution, lock file writing, and StateDB routing.

**Files:**
- Test: `tests/integration/test_all_pipelines.py` (new file)

**Step 1: Write the integration test**

Create `tests/integration/test_all_pipelines.py`:

```python
"""Integration tests for --all flag across multiple pipelines."""

from __future__ import annotations

import pathlib

import pytest
from click.testing import CliRunner

from conftest import isolated_pivot_dir
from helpers import create_pipeline_py
from pivot import cli
from pivot.storage import lock


def _helper_stage_a() -> None:
    """Stage for pipeline alpha."""
    pass


def _helper_stage_b() -> None:
    """Stage for pipeline beta."""
    pass


def _setup_multi_pipeline_project(
    root: pathlib.Path,
) -> None:
    """Create a project with two pipelines using different state_dirs."""
    # Alpha: uses project root as its root (shared .pivot)
    alpha = root / "alpha"
    alpha.mkdir()
    create_pipeline_py(
        [_helper_stage_a],
        path=alpha,
        names={"_helper_stage_a": "stage_a"},
        extra_code='pipeline = Pipeline("alpha", root=__import__("pathlib").Path(__file__).parent)\n',
    )

    # Beta: uses its own root (separate .pivot)
    beta = root / "beta"
    beta.mkdir()
    (beta / ".pivot").mkdir()
    create_pipeline_py(
        [_helper_stage_b],
        path=beta,
        names={"_helper_stage_b": "stage_b"},
        extra_code='pipeline = Pipeline("beta", root=__import__("pathlib").Path(__file__).parent)\n',
    )


def test_repro_all_executes_stages_from_all_pipelines(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --all runs stages from all discovered pipelines."""
    with isolated_pivot_dir(runner, tmp_path):
        _setup_multi_pipeline_project(tmp_path)

        result = runner.invoke(cli.cli, ["repro", "--all"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" in result.output


def test_repro_all_writes_locks_to_correct_state_dir(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """Lock files are written to each pipeline's own .pivot/stages/."""
    with isolated_pivot_dir(runner, tmp_path):
        _setup_multi_pipeline_project(tmp_path)

        runner.invoke(cli.cli, ["repro", "--all"])

        # Beta's lock file should be in beta/.pivot/stages/
        beta_lock = lock.StageLock("stage_b", lock.get_stages_dir(tmp_path / "beta" / ".pivot"))
        assert beta_lock.read() is not None, "Lock file not found in beta's state_dir"


def test_status_all_shows_all_pipelines(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """status --all shows stages from all pipelines."""
    with isolated_pivot_dir(runner, tmp_path):
        _setup_multi_pipeline_project(tmp_path)

        result = runner.invoke(cli.cli, ["status", "--all"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" in result.output


def test_verify_all_with_mixed_state_dirs(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """verify --all reads lock files from correct per-pipeline state_dirs."""
    with isolated_pivot_dir(runner, tmp_path):
        _setup_multi_pipeline_project(tmp_path)

        # Run first to create locks
        runner.invoke(cli.cli, ["repro", "--all"])

        result = runner.invoke(cli.cli, ["verify", "--all"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" in result.output
```

Note: The exact `create_pipeline_py` usage depends on the existing helper's interface. Adjust `extra_code` to match how pipelines are constructed in test helpers.

**Step 2: Run the integration tests**

Run: `uv run pytest tests/integration/test_all_pipelines.py -v`
Expected: All pass

---

### Task 10: Quality checks and smoke test

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

# Test each command
uv run pivot status --all
uv run pivot verify --all
uv run pivot repro --all --dry-run
uv run pivot repro --all
uv run pivot push --all --dry-run
```

Expected: All commands discover all 6 pipelines and operate correctly across shared and separate state_dirs.

---

## Summary of changes by file

| File | Change | Task |
|------|--------|------|
| `src/pivot/cli/repro.py:741` | `allow_all=True` | 1 |
| `src/pivot/cli/status.py:29` | `allow_all=True` | 1 |
| `src/pivot/cli/commit.py:12` | `allow_all=True` | 1 |
| `src/pivot/engine/engine.py:114` | Add `all_pipelines` param to `__init__` | 8 |
| `src/pivot/engine/engine.py:533,557,567` | StateDB connection cache per `state_dir` | 2 |
| `src/pivot/engine/engine.py:607-613` | Per-stage StateDB for deferred writes | 2 |
| `src/pivot/engine/engine.py:1081-1085` | Per-stage `state_dir` for lock file reads | 3 |
| `src/pivot/engine/engine.py:1301-1332` | Use `all_pipelines` flag in reload | 8 |
| `src/pivot/engine/engine.py:1186-1192` | Add pipeline config files to watch paths | 8 |
| `src/pivot/executor/core.py:448-477` | Per-stage `state_dir` for incremental check | 4 |
| `src/pivot/executor/commit.py:14-61` | Per-stage `state_dir` for production locks | 5 |
| `src/pivot/cli/checkout.py:35-64,336` | Per-stage `state_dir` for output restoration | 6 |
| `src/pivot/cli/remote.py:43,112,183` | `allow_all=True` on push/pull/fetch | 7 |
| `src/pivot/cli/decorators.py` | Discovery when `use_all` even if `auto_discover=False` | 7 |
| `src/pivot/remote/sync.py:132-185` | Per-stage `state_dir` for target hash resolution | 7 |
| `tests/integration/test_all_pipelines.py` | New: E2E tests for `--all` with mixed state_dirs | 9 |
