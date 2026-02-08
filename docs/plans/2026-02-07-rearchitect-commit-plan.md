# Rearchitect Commit Implementation Plan


**Goal:** Eliminate the "pending lock" concept, simplify `--no-commit` mode, drop `--no-cache`, and remove `None` from `OutputHash` throughout the codebase.

**Architecture:** (1) `--no-commit` hashes outputs but writes nothing durable, (2) `pivot repro` still commits by default, (3) `pivot commit` snapshots current workspace state using the registry and outputs on disk, (4) `OutputHash` becomes non-nullable (`FileHash | DirHash`).

**Tech Stack:** Python 3.13+, pytest, pivot internals

**Design doc:** `docs/plans/2026-02-07-rearchitect-commit-design.md`

---

### Task 1: Remove `OutputHash` type alias, make output hashes non-nullable

Remove `None` from the output hash union type and update all type annotations. This is the foundation for everything else.

**Files:**
- Modify: `src/pivot/types.py`
- Modify: `src/pivot/executor/worker.py`
- Modify: `src/pivot/storage/lock.py`
- Modify: `src/pivot/run_history.py`
- Modify: `src/pivot/cli/checkout.py`
- Modify: `src/pivot/cli/verify.py`
- Modify: `src/pivot/remote/sync.py`

**Step 1: Update types.py**

In `src/pivot/types.py`:

- Remove the `OutputHash` type alias (line 177: `OutputHash = FileHash | DirHash | None`)
- Update `LockData["output_hashes"]` from `dict[str, OutputHash]` to `dict[str, HashInfo]`
- Update `OutEntry["hash"]` from `str | None` to `str`
- Remove `OutputHash` from the module-level comment block (lines 148-151)

**Step 2: Update all imports of `OutputHash`**

Replace `OutputHash` imports with `HashInfo` in:
- `src/pivot/executor/worker.py` (line 41) — remove `OutputHash` from import, ensure `HashInfo` is imported
- `src/pivot/storage/lock.py` (line 25) — remove `OutputHash` from import
- `src/pivot/run_history.py` (line 15) — remove `OutputHash` import
- `src/pivot/cli/checkout.py` (line 16) — remove `OutputHash` import, use `HashInfo`
- `src/pivot/cli/verify.py` (line 19) — remove `OutputHash` import

**Step 3: Remove `None` checks on output hashes**

- `src/pivot/executor/worker.py` `_restore_outputs` (~line 501): remove the `if output_hash is None:` branch — all hashes are now non-null
- `src/pivot/storage/lock.py` `_convert_to_storage_format` (~line 86-87): remove `if hash_info is None:` branch
- `src/pivot/storage/lock.py` `_convert_from_storage_format` (~line 125-126): remove `if entry["hash"] is None:` branch
- `src/pivot/cli/checkout.py` `_checkout_files_async` (~line 152-154): remove `if output_hash is None:` skip
- `src/pivot/cli/verify.py` `_extract_file_hashes` (~line 54-55): remove `if hash_info is None: continue`

**Step 4: Simplify `output_hash_to_entry` in run_history.py**

`output_hash_to_entry` (line 55-62) currently returns `None` for uncached outputs. Change return type from `OutputHashEntry | None` to `OutputHashEntry` and remove the `None` check:

```python
def output_hash_to_entry(path: str, oh: HashInfo) -> OutputHashEntry:
    """Convert internal HashInfo to serializable OutputHashEntry."""
    entry = OutputHashEntry(path=path, hash=oh["hash"])
    if "manifest" in oh:
        entry["manifest"] = oh["manifest"]
    return entry
```

**Step 5: Update `write_run_cache_entry` in worker.py**

The function signature (~line 1134) takes `output_hashes: dict[str, OutputHash]` — change to `dict[str, HashInfo]`. The filter `if (entry := ...) is not None` in callers (`_build_deferred_writes` line 1040-1043 and `write_run_cache_entry` line 1139-1142) can be simplified since `output_hash_to_entry` no longer returns `None`.

**Step 6: Run tests and fix any type errors**

```bash
uv run basedpyright
uv run pytest tests/ -x -n auto
```

Expect some test failures in `tests/execution/test_execution_modes.py` (the `--no-cache` tests assert `None` hashes) — those tests will be removed in Task 2.

**Step 7: Commit**

---

### Task 2: Remove `--no-cache` flag and all associated code

**Files:**
- Modify: `src/pivot/executor/worker.py` — remove `no_cache` paths, `_verify_outputs_exist`
- Modify: `src/pivot/executor/core.py` — remove `no_cache` parameter
- Modify: `src/pivot/engine/engine.py` — remove `no_cache` plumbing
- Modify: `src/pivot/engine/types.py` — remove `no_cache` from `RunRequested`
- Modify: `src/pivot/engine/sources.py` — remove `no_cache` from `create_run_event`
- Modify: `src/pivot/engine/agent_rpc.py` — remove `no_cache` default
- Modify: `src/pivot/cli/run.py` — remove `--no-cache` option and plumbing
- Modify: `src/pivot/cli/repro.py` — remove `--no-cache` option and plumbing
- Delete tests: `tests/execution/test_execution_modes.py` functions `test_no_cache_*` (4-5 tests)

**Step 1: Remove from worker**

In `src/pivot/executor/worker.py`:

- `WorkerStageInfo` (line 142): remove `no_cache: bool`
- `execute_stage` (line 194): remove `no_cache = stage_info["no_cache"]`
- Remove IncrementalOut incompatibility check (lines 197-203)
- Remove the `if no_cache:` branch in output hashing (lines 335-339) — always call `_save_outputs_to_cache`
- Delete `_verify_outputs_exist` function (lines 654-663)

**Step 2: Remove from executor/core.py**

In `src/pivot/executor/core.py`:

- `run()` function: remove `no_cache` parameter (~line 161)
- `_run_impl()`: remove `no_cache` parameter (~line 222)
- `_build_worker_stage_info()`: remove `no_cache` parameter (~line 324) and `no_cache=no_cache` from the dict construction (~line 351)

**Step 3: Remove from engine layer**

- `src/pivot/engine/types.py` line 100: remove `no_cache: bool` from `RunRequested`
- `src/pivot/engine/engine.py`: remove all `no_cache` references (lines 105, 149, 345, 369, 457, 576, 685, 904, 946, 1215)
- `src/pivot/engine/sources.py`: remove `no_cache` parameter (~line 68) and from dict construction (~line 99)
- `src/pivot/engine/agent_rpc.py`: remove `no_cache=False` (~line 450)

**Step 4: Remove from CLI**

- `src/pivot/cli/repro.py`: remove `--no-cache` option (~line 788), remove `no_cache` from all function signatures and call sites (many locations — search for `no_cache`)
- `src/pivot/cli/run.py`: remove `--no-cache` option (~line 353), remove `no_cache` from all function signatures and call sites

**Step 5: Remove tests**

In `tests/execution/test_execution_modes.py`, delete:
- `test_no_cache_skips_cache_operations` (~line 390)
- `test_no_cache_writes_lock_with_null_hashes` (~line 420)
- `test_no_cache_second_run_still_skips` (~line 452)
- `test_no_cache_incompatible_with_incremental_out` (~line 483)
- `test_no_cache_with_no_commit` (~line 506)

Update the module docstring (line 1) from `"""Tests for execution modes: --no-commit, --no-cache, and commit command."""` to `"""Tests for execution modes: --no-commit and commit command."""`.

Update test helpers that pass `no_cache` (e.g., `_make_stage_info` at ~line 34-35):
- Remove `no_cache: bool = False` parameter
- Remove `no_cache=no_cache` from the WorkerStageInfo construction

Also check and fix:
- `tests/execution/test_executor_worker.py` (~line 71): remove `"no_cache": no_cache` or `"no_cache": False`
- `tests/test_run_cache_lock_update.py` (~line 56): remove `"no_cache": False`
- `tests/cli/test_cli_completion.py`: remove `--no-cache` from completion expectations
- `tests/cli/test_repro.py`, `tests/cli/test_run.py`: remove any `--no-cache` test invocations
- `tests/engine/test_engine.py`, `tests/engine/test_types.py`: remove `no_cache` from event constructions

**Step 6: Run tests and quality checks**

```bash
uv run basedpyright
uv run ruff check .
uv run pytest tests/ -x -n auto
```

**Step 7: Commit**

---

### Task 3: Propagate `input_hash` through `StageResult`

**Files:**
- Modify: `src/pivot/types.py` — add `input_hash` to `StageResult`
- Modify: `src/pivot/executor/worker.py` — set `input_hash` in all result constructions
- Modify: `src/pivot/engine/engine.py` — read `input_hash` from result in `_write_run_history`
- Modify: `src/pivot/run_history.py` — remove `compute_input_hash_from_lock`, update `StageRunRecord`

**Step 1: Add `input_hash` to `StageResult`**

In `src/pivot/types.py`, add to `StageResult` (after `reason: str`):

```python
input_hash: str | None  # None only for early failures before dep hashing
```

Update `StageRunRecord["input_hash"]` from `str` to `str | None`.

**Step 2: Update `_make_result` in worker.py**

Add `input_hash: str | None = None` parameter to `_make_result` (~line 150):

```python
def _make_result(
    status: Literal[StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED],
    reason: str,
    output_lines: list[tuple[str, bool]],
    input_hash: str | None = None,
) -> StageResult:
    """Build StageResult with collected metrics for cross-process transfer."""
    return StageResult(
        status=status,
        reason=reason,
        output_lines=output_lines,
        metrics=metrics.get_entries(),
        input_hash=input_hash,
    )
```

**Step 3: Set `input_hash` in all call sites**

In `execute_stage()`:

- Early failures BEFORE `_check_skip_or_run` (params validation ~219, missing deps ~238, unreadable deps ~243): pass `input_hash=None`
- All calls AFTER `_check_skip_or_run` (which returns `input_hash`): pass `input_hash=input_hash`
  - Skip via generation/lock (~272): `_make_result(StageStatus.SKIPPED, skip_reason, [], input_hash=input_hash)`
  - Successful run (~375): `_make_result(StageStatus.RAN, run_reason, output_lines, input_hash=input_hash)`
  - All exception handlers (~378, 380, 382, 386, 388): pass `input_hash=input_hash` (available since `_check_skip_or_run` ran before execution)
- Direct `StageResult(...)` constructions:
  - IncrementalOut incompatibility (~198): `input_hash=None` (this was removed in Task 2 — skip)
  - Run cache skip (~304): add `input_hash=input_hash`
  - Deferred writes RAN (~367): add `input_hash=input_hash`
- `_make_result` inside `_try_skip_via_run_cache` (~1126): this is inside `RunCacheSkipResult`, the `input_hash` comes from the caller — add `input_hash` parameter to `_try_skip_via_run_cache` and thread it through.

**Step 4: Update `_write_run_history` in engine.py**

In `_write_run_history` (~line 1066), the method receives `results: dict[str, executor_core.ExecutionSummary]`. The `ExecutionSummary` type needs to carry `input_hash`. Check `executor_core.ExecutionSummary` — it's a simplified view. The `input_hash` needs to be propagated from `StageResult` → `ExecutionSummary` → `_write_run_history`.

Check how `ExecutionSummary` is built from `StageResult` in `core.py` and add `input_hash: str | None` to `ExecutionSummary`. Then in `_write_run_history`, replace:

```python
# Before:
stage_lock = lock.StageLock(name, lock.get_stages_dir(state_dir))
lock_data = stage_lock.read()
if lock_data:
    input_hash = run_history.compute_input_hash_from_lock(lock_data)
else:
    input_hash = "<no-lock>"

# After:
input_hash = summary.get("input_hash")  # Already propagated from worker
```

**Step 5: Remove `compute_input_hash_from_lock`**

Delete `compute_input_hash_from_lock` function from `src/pivot/run_history.py` (lines 102-117).

Remove its import from `src/pivot/executor/commit.py` and `src/pivot/engine/engine.py`.

**Step 6: Run tests**

```bash
uv run basedpyright
uv run pytest tests/ -x -n auto
```

**Step 7: Commit**

---

### Task 4: Remove pending lock infrastructure

**Files:**
- Modify: `src/pivot/storage/lock.py` — remove pending functions and constant
- Modify: `src/pivot/storage/project_lock.py` — remove `pending_state_lock`
- Modify: `src/pivot/executor/worker.py` — remove pending lock reads/writes
- Modify: `src/pivot/executor/commit.py` — gut existing functions
- Modify: `src/pivot/cli/commit.py` — remove `--list`, `--discard`, pending_state_lock

**Step 1: Remove pending infrastructure from lock.py**

In `src/pivot/storage/lock.py`, delete:
- `_PENDING_DIR = "pending"` (line 38)
- `get_pending_stages_dir()` (lines 51-53)
- `get_pending_lock()` (lines 230-232)
- `list_pending_stages()` (lines 235-240)

**Step 2: Remove `pending_state_lock` from project_lock.py**

In `src/pivot/storage/project_lock.py`, delete:
- `_PENDING_LOCK_NAME` constant (~line 17)
- `pending_state_lock()` context manager (~lines 37-54)
- `acquire_pending_state_lock()` (~lines 57-73)

**Step 3: Remove pending lock usage from worker**

In `src/pivot/executor/worker.py` `execute_stage()`:
- Remove `pending_lock = lock.get_pending_lock(stage_name, project_root)` (~line 208)
- Remove `pending_lock_data = pending_lock.read()` (~line 228)
- Change `lock_data = pending_lock_data or production_lock_data` (~line 230) to `lock_data = production_lock_data`
- Remove `pending_lock` from `_commit_lock_and_build_deferred` call sites (~lines 298, 356)
- Remove `pending_lock` parameter from `_commit_lock_and_build_deferred` function signature (~line 1004)

In `_commit_lock_and_build_deferred`:
- Remove the `pending_lock` parameter
- The `no_commit` branch currently writes `pending_lock.write(lock_data)` — change to return empty `DeferredWrites` (noop):

```python
if no_commit:
    return DeferredWrites()
```

(Note: `DeferredWrites` uses `total=False`, so an empty dict is valid.)

- Also remove `from pivot.storage import lock` import if `pending_lock` was the only reason (check — likely still needed for `production_lock`)

**Step 4: Gut commit.py**

In `src/pivot/executor/commit.py`:
- Delete `COMMITTED_RUN_ID` sentinel
- Delete `commit_pending()` function entirely
- Delete `discard_pending()` function entirely
- Leave the file as a stub (the new `commit_stages()` will be added in Task 6)

**Step 5: Simplify cli/commit.py**

Rewrite `src/pivot/cli/commit.py` as a minimal stub that will be filled in Task 6:

```python
from __future__ import annotations

import click

from pivot.cli import decorators as cli_decorators


@cli_decorators.pivot_command("commit")
@click.argument("stages", nargs=-1)
@click.pass_context
def commit_command(ctx: click.Context, stages: tuple[str, ...]) -> None:
    """Commit current workspace state for stages.

    Hashes current deps and outputs, writes lock files and cache.
    Without arguments, commits all stale stages.
    """
    click.echo("pivot commit: not yet implemented (rearchitect in progress)")
```

**Step 6: Remove pending-related tests**

Delete or update tests that test pending behavior:
- `tests/cli/test_cli_commit.py` — gut entirely (rewrite in Task 7)
- `tests/execution/test_execution_modes.py` — remove tests that reference pending locks (check `test_no_commit_*` tests)
- `tests/storage/test_project_lock.py` — remove pending_state_lock tests

Search for `pending` in test files and update each reference.

**Step 7: Run tests**

```bash
uv run basedpyright
uv run pytest tests/ -x -n auto
```

**Step 8: Commit**

---

### Task 5: Rearchitect `--no-commit` in the worker

Now that pending infrastructure is gone, simplify the `--no-commit` worker path.

**Files:**
- Modify: `src/pivot/executor/worker.py`

**Step 1: Simplify output hashing for `--no-commit`**

In `execute_stage()`, the output hashing section (~lines 334-343) currently has `if no_cache:` / `else:` branches. After Task 2 removed `--no-cache`, this is just `_save_outputs_to_cache`. Add the `no_commit` branch:

```python
if no_commit:
    output_hashes = _hash_outputs_only(stage_outs)
else:
    output_hashes = _save_outputs_to_cache(stage_outs, files_cache_dir, checkout_modes)
```

Add the helper:

```python
def _hash_outputs_only(stage_outs: list[outputs.BaseOut]) -> dict[str, HashInfo]:
    """Hash outputs without saving to cache (for --no-commit mode)."""
    output_hashes = dict[str, HashInfo]()
    for out in stage_outs:
        path = pathlib.Path(cast("str", out.path))
        if not path.exists():
            raise exceptions.OutputMissingError(f"Stage did not produce output: {out.path}")
        output_hashes[str(out.path)] = _hash_output(path)
    return output_hashes
```

**Step 2: Simplify `_commit_lock_and_build_deferred`**

The `no_commit` branch should return empty deferred writes (already done in Task 4). Verify it looks like:

```python
if no_commit:
    return DeferredWrites()
production_lock.write(lock_data)
return _build_deferred_writes(stage_info, input_hash, output_hashes, state_db)
```

**Step 3: Clean up the second StateDB open**

In `execute_stage`, after the stage runs, there's a `with state.StateDB(state_db_path, readonly=True) as state_db:` block (~line 355) for post-execution work. For `no_commit`, this StateDB open is unnecessary since we're not writing deferred writes. Add early return:

```python
if no_commit:
    new_lock_data = LockData(...)  # still compute for StageResult
    return StageResult(
        status=StageStatus.RAN,
        reason=run_reason,
        output_lines=output_lines,
        metrics=metrics.get_entries(),
        input_hash=input_hash,
    )
```

This avoids opening StateDB unnecessarily in `--no-commit` mode.

**Step 4: Run tests**

```bash
uv run pytest tests/ -x -n auto
uv run basedpyright
```

**Step 5: Commit**

---

### Task 6: Implement new `pivot commit` command

The core new functionality. `pivot commit` computes from current workspace state.

**Files:**
- Rewrite: `src/pivot/executor/commit.py` — new `commit_stages()` function
- Rewrite: `src/pivot/cli/commit.py` — new CLI command

**Step 1: Write the core `commit_stages()` function**

Rewrite `src/pivot/executor/commit.py`:

```python
from __future__ import annotations

import logging
import pathlib
from typing import cast

from pivot import config, exceptions, fingerprint, outputs, parameters, project, run_history
from pivot.executor import worker
from pivot.storage import cache, lock
from pivot.storage import state as state_mod
from pivot.types import DepEntry, HashInfo, LockData

logger = logging.getLogger(__name__)


def commit_stages(
    stage_names: list[str] | None = None,
    *,
    force: bool = False,
) -> list[str]:
    """Commit stages by snapshotting current workspace state.

    Computes fingerprints, hashes deps and outputs, writes production locks
    and updates StateDB. This is the "trust me" path — it records current
    filesystem state without re-running stages.

    Args:
        stage_names: Specific stages to commit. None = all stale stages.
        force: If True, commit even if production lock matches current state.

    Returns list of stage names that were committed.
    """
    from pivot import registry as registry_mod

    project_root = project.get_project_root()
    state_dir = config.get_state_dir()
    stages_dir = lock.get_stages_dir(state_dir)
    cache_dir = config.get_cache_dir()
    files_cache_dir = cache_dir / "files"

    # Determine target stages
    all_stages = registry_mod.list_stages()
    if stage_names is not None:
        # Validate requested stages exist
        unknown = set(stage_names) - set(all_stages)
        if unknown:
            raise exceptions.ValidationError(f"Unknown stages: {', '.join(sorted(unknown))}")
        targets = stage_names
    else:
        targets = all_stages

    committed = list[str]()

    with state_mod.StateDB(config.get_state_db_path()) as state_db:
        for stage_name in targets:
            stage_info = registry_mod.get_stage(stage_name)

            # Compute current state
            current_fingerprint = fingerprint.get_stage_fingerprint_cached(
                stage_name, stage_info["func"]
            )
            for spec in stage_info["dep_specs"].values():
                current_fingerprint.update(fingerprint.get_loader_fingerprint(spec.loader))
            for out in stage_info["out_specs"].values():
                current_fingerprint.update(fingerprint.get_loader_fingerprint(out.loader))

            current_params = parameters.get_effective_params(
                stage_info.get("params"), stage_name, {}
            )

            # Hash dependencies
            dep_paths = _get_dep_paths(stage_info)
            dep_hashes, missing, unreadable = worker.hash_dependencies(dep_paths, state_db)
            if missing:
                logger.warning("Stage '%s': missing deps: %s", stage_name, ", ".join(missing))
                continue
            if unreadable:
                logger.warning("Stage '%s': unreadable deps: %s", stage_name, ", ".join(unreadable))
                continue

            # Compute input hash
            deps_list = [
                DepEntry(path=path, hash=info["hash"]) for path, info in dep_hashes.items()
            ]
            outs_list = _get_expanded_outs(stage_info)
            out_specs = [(str(out.path), out.cache) for out in outs_list]
            input_hash = run_history.compute_input_hash(
                current_fingerprint, current_params, deps_list, out_specs
            )

            # Check if commit is needed (unless force or explicitly targeted)
            if not force and stage_names is None:
                production_lock = lock.StageLock(stage_name, stages_dir)
                changed, _ = production_lock.is_changed(
                    current_fingerprint, current_params, dep_hashes
                )
                if not changed:
                    continue

            # Hash and cache outputs
            output_hashes = dict[str, HashInfo]()
            all_exist = True
            for out in outs_list:
                path = pathlib.Path(cast("str", out.path))
                if not path.exists():
                    logger.error("Stage '%s': missing output: %s", stage_name, out.path)
                    all_exist = False
                    break
                if out.cache:
                    output_hashes[str(out.path)] = cache.save_to_cache(
                        path, files_cache_dir, checkout_modes=[cache.CheckoutMode.COPY]
                    )
                else:
                    output_hashes[str(out.path)] = worker._hash_output(path, state_db)

            if not all_exist:
                continue

            # Write production lock
            new_lock_data = LockData(
                code_manifest=current_fingerprint,
                params=current_params,
                dep_hashes=dict(sorted(dep_hashes.items())),
                output_hashes=dict(sorted(output_hashes.items())),
                dep_generations={},
            )
            production_lock = lock.StageLock(stage_name, stages_dir)
            production_lock.write(new_lock_data)

            # Update StateDB
            dep_gens = worker.compute_dep_generation_map(dep_paths, state_db)
            if dep_gens:
                state_db.record_dep_generations(stage_name, dep_gens)

            for out_path in output_hashes:
                state_db.increment_generation(pathlib.Path(out_path))

            # Write run cache entry
            worker.write_run_cache_entry(
                stage_name, input_hash, output_hashes, run_history.generate_run_id(), state_db
            )

            committed.append(stage_name)

    return committed


def _get_dep_paths(stage_info: dict) -> list[str]:
    """Get absolute dep paths from stage info for hashing."""
    # Reuse the same dep resolution that the executor uses
    from pivot import registry as registry_mod

    return registry_mod.get_stage_deps(stage_info)


def _get_expanded_outs(stage_info: dict) -> list[outputs.BaseOut]:
    """Get expanded output specs from stage info."""
    from pivot import registry as registry_mod

    return registry_mod.get_stage_outs(stage_info)
```

Note: The exact helper functions (`_get_dep_paths`, `_get_expanded_outs`) depend on what the registry exposes. Check `registry.py` for the right API — the executor's `_build_worker_stage_info` in `core.py` shows the pattern. This code is a sketch; adjust to match the actual registry API.

**Step 2: Write the CLI command**

Rewrite `src/pivot/cli/commit.py`:

```python
from __future__ import annotations

import click

from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.executor import commit


@cli_decorators.pivot_command("commit")
@click.argument("stages", nargs=-1)
@click.pass_context
def commit_command(ctx: click.Context, stages: tuple[str, ...]) -> None:
    """Commit current workspace state for stages.

    Hashes current deps and outputs, writes lock files and cache.
    Without arguments, commits all stale stages.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    stage_names = list(stages) if stages else None
    committed = commit.commit_stages(stage_names)

    if not quiet:
        if not committed:
            click.echo("Nothing to commit")
        else:
            click.echo(f"Committed {len(committed)} stage(s):")
            for stage_name in committed:
                click.echo(f"  {stage_name}")
```

**Step 3: Run tests**

```bash
uv run basedpyright
uv run pytest tests/ -x -n auto
```

**Step 4: Commit**

---

### Task 7: Remote sync filtering

Replace `None`-based filtering with registry-based filtering.

**Files:**
- Modify: `src/pivot/remote/sync.py`

**Step 1: Update `get_stage_output_hashes`**

In `get_stage_output_hashes()` (~line 53-68), filter non-cached outputs using the registry:

```python
def get_stage_output_hashes(stage_names: list[str], state_dir: Path) -> set[str]:
    """Get content hashes for cached stage outputs (for push/pull)."""
    from pivot.cli import helpers as cli_helpers

    hashes = set[str]()

    for stage_name in stage_names:
        stage_info = cli_helpers.get_stage(stage_name)
        non_cached_paths = {str(out.path) for out in stage_info["outs"] if not out.cache}

        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        lock_data = stage_lock.read()
        if lock_data is None:
            logger.warning(f"No lock file for stage '{stage_name}'")
            continue

        for out_path, output_hash in lock_data["output_hashes"].items():
            if out_path not in non_cached_paths:
                hashes |= _extract_hashes_from_hash_info(output_hash)

    return hashes
```

**Step 2: Update `_get_file_hash_from_stages`**

In `_get_file_hash_from_stages()` (~line 96-114), same pattern — filter by registry:

```python
def _get_file_hash_from_stages(rel_path: str, state_dir: Path) -> HashInfo | None:
    from pivot.cli import helpers as cli_helpers

    stages_dir = lock.get_stages_dir(state_dir)

    for lock_file in stages_dir.glob("*.lock"):
        stage_name = lock_file.stem
        try:
            stage_info = cli_helpers.get_stage(stage_name)
        except Exception:
            continue
        non_cached_paths = {str(out.path) for out in stage_info["outs"] if not out.cache}

        stage_lock = lock.StageLock(stage_name, stages_dir)
        lock_data = stage_lock.read()
        if lock_data is None:
            continue

        for out_path, out_hash in lock_data["output_hashes"].items():
            if out_path == rel_path and out_path not in non_cached_paths:
                return out_hash

    return None
```

**Step 3: Update `get_target_hashes`**

In `get_target_hashes()` (~line 144-180), apply the same filter when resolving stage name targets:

```python
# Inside the stage_name branch:
stage_info = cli_helpers.get_stage(target)
non_cached_paths = {str(out.path) for out in stage_info["outs"] if not out.cache}
for out_path, out_hash in lock_data["output_hashes"].items():
    if out_path not in non_cached_paths:
        hashes |= _extract_hashes_from_hash_info(out_hash)
```

**Step 4: Run tests**

```bash
uv run basedpyright
uv run pytest tests/ -x -n auto
```

**Step 5: Commit**

---

### Task 8: Write tests for new `pivot commit` behavior

**Files:**
- Rewrite: `tests/cli/test_cli_commit.py`
- Update: `tests/execution/test_execution_modes.py` — update `--no-commit` tests

**Step 1: Write `pivot commit` tests**

Rewrite `tests/cli/test_cli_commit.py` with tests covering:

1. **`pivot commit` with no args commits stale stages** — run a stage, modify code, verify `pivot commit` updates the lock file
2. **`pivot commit <stage>` unconditionally commits** — run a stage, `pivot commit <stage>` even when lock matches
3. **`pivot commit` with missing outputs errors** — register a stage, don't run it, `pivot commit` should error for that stage
4. **`pivot commit` skips unchanged stages** — run a stage normally (committed), `pivot commit` with no args should skip it
5. **`pivot commit` after `--no-commit` run** — run with `--no-commit`, verify `pivot commit` writes lock and caches outputs

**Step 2: Update `--no-commit` tests**

In `tests/execution/test_execution_modes.py`, update the `test_no_commit_*` tests:
- Remove assertions about pending lock files
- Verify that `--no-commit` produces outputs on disk but no production lock
- Verify that a subsequent normal `pivot repro` re-runs and commits

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -x -n auto
```

**Step 4: Commit**

---

### Task 9: Final quality checks and cleanup

**Step 1: Run full quality suite**

```bash
uv run ruff format .
uv run ruff check .
uv run basedpyright
uv run pytest tests/ -x -n auto
```

**Step 2: Search for stale references**

```bash
# Should find nothing:
rg "OutputHash" src/
rg "compute_input_hash_from_lock" src/
rg "pending_state_lock" src/
rg "commit_pending\|discard_pending" src/
rg "COMMITTED_RUN_ID" src/
rg "_PENDING_DIR\|get_pending" src/
rg "no_cache" src/pivot/
```

**Step 3: Verify docs**

Check if `docs/architecture/execution.md` references pending locks or `--no-cache` — update if needed.

**Step 4: Commit**

---

## Files Modified Summary

| File | Change |
|------|--------|
| `src/pivot/types.py` | Remove `OutputHash`, add `input_hash` to `StageResult` and `StageRunRecord` |
| `src/pivot/executor/worker.py` | Remove `no_cache`, pending locks; add `_hash_outputs_only`; propagate `input_hash` |
| `src/pivot/executor/core.py` | Remove `no_cache` parameter |
| `src/pivot/executor/commit.py` | Rewrite: new `commit_stages()` |
| `src/pivot/engine/engine.py` | Remove `no_cache`, use `input_hash` from `StageResult` |
| `src/pivot/engine/types.py` | Remove `no_cache` from `RunRequested` |
| `src/pivot/engine/sources.py` | Remove `no_cache` |
| `src/pivot/engine/agent_rpc.py` | Remove `no_cache` |
| `src/pivot/storage/lock.py` | Remove pending infrastructure, update `OutEntry` |
| `src/pivot/storage/project_lock.py` | Remove `pending_state_lock` |
| `src/pivot/run_history.py` | Remove `compute_input_hash_from_lock`, simplify `output_hash_to_entry` |
| `src/pivot/cli/commit.py` | Rewrite: new command with stage args |
| `src/pivot/cli/run.py` | Remove `--no-cache` |
| `src/pivot/cli/repro.py` | Remove `--no-cache` |
| `src/pivot/cli/checkout.py` | Remove `None` checks |
| `src/pivot/cli/verify.py` | Remove `None` checks |
| `src/pivot/remote/sync.py` | Registry-based filtering |
| `tests/cli/test_cli_commit.py` | Rewrite for new behavior |
| `tests/execution/test_execution_modes.py` | Remove `--no-cache` tests, update `--no-commit` tests |
| Multiple test files | Remove `no_cache` from fixtures |
