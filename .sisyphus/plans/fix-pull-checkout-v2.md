# Fix pull --all checkout crash and refine error handling

## TL;DR

> **Quick Summary**: `pivot pull --all` crashes with `TypeError: object of type 'NoneType' has no len()` during the checkout phase because `_get_stage_output_info()` reads lock files from all 172 stages — some have never been executed and contain null hash values. Fix by skipping entries with null hashes, and add verbose traceback logging for future debugging.
> 
> **Deliverables**:
> - `pivot pull --all` completes checkout without crashing on null-hash lock entries
> - Null-hash entries logged as warnings (not silent)
> - `--verbose` mode shows full tracebacks for unhandled exceptions
> - Tests for all new behaviors
> 
> **Estimated Effort**: Short
> **Parallel Execution**: NO - sequential (each task builds on the previous)
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
Running `pivot pull --all` in a project with sub-pipelines (6 pipelines, 172 stages) crashes:
```
Discovered 6 pipelines with 172 total stages
Fetched from 'sami': 0 transferred, 560 skipped, 0 failed
Error: object of type 'NoneType' has no len()
object of type 'NoneType' has no len()
```

The fetch phase succeeds but the checkout phase crashes when trying to restore stage outputs from lock files that have null hash values (stages that were never executed locally).

### Interview Summary
**Key Discussions**:
- `.pvt` files are dependencies of the pipeline DAG — they're pulled as part of the pipeline, not independently
- `pull` and `checkout` should leave you ready to `repro` whatever pipeline you're targeting
- `pull --all` should work across all pipelines and all their `.pvt` dependencies
- No pipeline = error (not fall back to .pvt files independently)

**Research Findings**:
- `cache.get_cache_path()` at `cache.py:220` calls `len(file_hash)` — when `output_hash["hash"]` is None, this crashes with `TypeError: object of type 'NoneType' has no len()`
- `_get_stage_output_info()` iterates ALL stages in the pipeline, reads their lock files, and collects output hashes. Stages that were never executed locally have incomplete lock data with null hashes.
- `_checkout_files_async`'s `except* Exception` handler (line 156-159) collects multiple TypeErrors, converts them to strings, and joins with newlines — explaining the doubled error output (2 stages had null hashes)
- `ctx.invoke()` shares the same Click context — checkout inherits the multi-pipeline Pipeline from pull's context. Pipeline discovery is NOT re-run.
- `discover_pvt_files()` uses `root.rglob("*.pvt")` independently of pipeline context.
- `checkout` doesn't have `allow_all=True` but correctly inherits the multi-pipeline from pull's context via `_has_pipeline_in_context()` check.

### Metis Review
**Identified Gaps** (addressed):
- Silently skipping null hashes could mask real corruption → use `logger.warning()` so it's visible in verbose mode
- Lockfiles with missing keys (no "hash" key at all) or empty string hash → guard against both
- Multiple pipelines with mixed lockfile health → checkout is best-effort (skip bad entries, restore what's available)
- Adding verbose traceback could expose sensitive paths → gate behind `--verbose` flag

---

## Work Objectives

### Core Objective
Fix `pivot pull --all` checkout crash by gracefully handling null/invalid hash entries in lock files, and add verbose traceback logging for future debugging.

### Concrete Deliverables
- Modified `src/pivot/cli/checkout.py`: `_get_stage_output_info()` skips lock entries with null/empty hashes
- Modified `src/pivot/cli/decorators.py`: `with_error_handling` logs full traceback in verbose mode
- New tests covering null-hash scenarios and verbose traceback behavior

### Definition of Done
- [x] `pivot pull --all` completes checkout without crashing on null-hash lock entries
- [x] Warning logged for each skipped null-hash entry
- [x] `--verbose` mode shows full tracebacks for unhandled exceptions
- [x] All existing tests still pass
- [x] New tests cover all changed behaviors

### Must Have
- Null/empty hash entries in lock files are skipped with a warning
- Checkout continues best-effort (restores what it can, skips broken entries)
- Verbose traceback logging for unhandled exceptions in CLI error handler
- No behavioral regression for normal (non-null-hash) checkout/pull

### Must NOT Have (Guardrails)
- Do NOT change caching semantics or lock file format
- Do NOT change pipeline discovery behavior
- Do NOT change fetch behavior (fetch already works correctly)
- Do NOT add new CLI flags or options
- Do NOT refactor the checkout TaskGroup or pipeline resolution broadly
- Do NOT expand `.pvt` discovery beyond current `rglob("*.pvt")` pattern
- Do NOT treat null hashes as fatal errors — checkout should be best-effort

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: YES (Tests-after — these are bug fixes)
- **Framework**: pytest (`uv run pytest`)

### Agent-Executed QA Scenarios (MANDATORY)

```
Scenario: pull --all handles stages with null-hash lock entries
  Tool: Bash (pytest)
  Preconditions: Test environment set up
  Steps:
    1. uv run pytest tests/cli/test_cli_checkout.py::test_checkout_skips_null_hash_in_stage_output -xvs
    2. Assert: exit code 0
    3. Assert: test passes
  Expected Result: Test passes
  Evidence: Terminal output captured

Scenario: verbose mode shows traceback on unhandled exception
  Tool: Bash (pytest)
  Preconditions: Test environment set up
  Steps:
    1. uv run pytest tests/cli/test_cli_checkout.py::test_verbose_traceback_on_error -xvs
    2. Assert: exit code 0
  Expected Result: Test passes
  Evidence: Terminal output captured

Scenario: all existing tests still pass
  Tool: Bash (pytest)
  Steps:
    1. uv run pytest tests/cli/test_cli_checkout.py tests/remote/test_cli_remote.py -x
    2. Assert: exit code 0
  Expected Result: All existing + new tests pass
  Evidence: Terminal output captured

Scenario: full quality checks pass
  Tool: Bash
  Steps:
    1. uv run ruff format . && uv run ruff check . && uv run basedpyright
    2. Assert: exit code 0
  Expected Result: No format/lint/type errors
  Evidence: Terminal output captured
```

---

## Execution Strategy

### Sequential Execution

```
Task 1: Fix _get_stage_output_info() to skip null/empty hashes
  ↓
Task 2: Add verbose traceback logging to error handler
  ↓
Task 3: Add tests for all new behaviors
  ↓
Task 4: Run full quality checks
```

---

## TODOs

- [x] 1. Fix `_get_stage_output_info()` to skip null/empty hash entries

  **What to do**:
  - In `src/pivot/cli/checkout.py`, modify `_get_stage_output_info()` (lines 35-67) to guard against null/empty hashes in lock file data
  - In the loop at line 60 (`for out_path, out_hash in lock_data["output_hashes"].items()`), add a check:
    - If `out_hash` is missing the "hash" key, or if `out_hash["hash"]` is None or empty string `""`, log a warning and `continue` (skip this entry)
  - Use `logger.warning(f"Skipping output '{out_path}' in stage '{stage_name}': invalid hash")` for the warning
  - This ensures checkout is best-effort: it restores what it can and skips broken entries

  **Must NOT do**:
  - Do not change `lock.StageLock.read()` behavior
  - Do not change `cache.get_cache_path()` — it should still validate its inputs
  - Do not silently skip (must log a warning)
  - Do not treat this as a fatal error — checkout should continue

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file change, ~5 lines of code, clear fix pattern
  - **Skills**: [`git-master`]
    - `git-master`: Need to commit changes

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 2, 3, 4
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/pivot/cli/checkout.py:35-67` — `_get_stage_output_info()` function, the crash site
  - `src/pivot/cli/checkout.py:59-65` — The inner loop that reads `lock_data["output_hashes"]` — guard goes here
  - `src/pivot/storage/cache.py:218-223` — `get_cache_path()` which crashes on `len(None)` — this is the downstream crash site we're preventing

  **API/Type References**:
  - `src/pivot/types.py:161-183` — `FileHash`, `DirHash`, `HashInfo` TypedDicts — all have `hash: str` field
  - `src/pivot/storage/lock.py:85-136` — `StageLock.read()` which reads and parses lock YAML — hashes come from here

  **Acceptance Criteria**:
  - [ ] `_get_stage_output_info()` skips lock entries where hash is None or empty
  - [ ] Warning logged for each skipped entry (visible in verbose mode via logger.warning)
  - [ ] When all hashes are valid: behavior unchanged
  - [ ] When some hashes are null: valid entries still collected, null entries skipped
  - [ ] Verification: `uv run pytest tests/cli/test_cli_checkout.py -x` passes (no regression)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: checkout still works with valid lock files (regression check)
    Tool: Bash (pytest)
    Steps:
      1. uv run pytest tests/cli/test_cli_checkout.py::test_checkout_stage_output -xvs
      2. uv run pytest tests/cli/test_cli_checkout.py::test_checkout_tracked_file -xvs
    Expected Result: Both pass (no regression)
  ```

  **Commit**: YES
  - Message: `fix(checkout): skip null-hash entries in stage output info`
  - Files: `src/pivot/cli/checkout.py`
  - Pre-commit: `uv run pytest tests/cli/test_cli_checkout.py -x`

---

- [x] 2. Add verbose traceback logging to CLI error handler

  **What to do**:
  - In `src/pivot/cli/decorators.py`, modify the `with_error_handling` wrapper (lines 42-53) to log the full traceback when in verbose mode before wrapping the exception
  - In the `except Exception as e:` handler at line 50-51:
    - Import `traceback` and `logging`
    - Get the verbose flag from Click context: check `ctx.obj` for verbose setting
    - If verbose: `logger.debug("Unhandled exception in CLI command", exc_info=True)` — this uses Python's built-in traceback formatting
    - Keep the existing `raise click.ClickException(repr(e)) from e` behavior
  - Alternative simpler approach: just always log at DEBUG level (which is only visible when `--verbose` is set since `_setup_logging` sets DEBUG level for verbose)

  **Must NOT do**:
  - Do not change the error handling behavior (still wrap as ClickException)
  - Do not print tracebacks by default (only in verbose/debug mode)
  - Do not change how PivotError is handled (line 48-49)
  - Do not add new CLI flags

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file change, ~3 lines of code
  - **Skills**: [`git-master`]
    - `git-master`: Need to commit changes

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 3, 4
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/pivot/cli/decorators.py:38-53` — `with_error_handling` function
  - `src/pivot/cli/decorators.py:50-51` — The `except Exception` handler that loses the traceback
  - `src/pivot/cli/__init__.py:140-148` — `_setup_logging()` which sets DEBUG level for `--verbose`

  **API/Type References**:
  - `src/pivot/cli/__init__.py:84` — `CliContext` TypedDict with `verbose: bool` field
  - `src/pivot/cli/helpers.py:95-106` — `get_cli_context()` which retrieves the context dict

  **Acceptance Criteria**:
  - [ ] `with_error_handling` logs full traceback at DEBUG level before wrapping exception
  - [ ] Traceback visible when `pivot --verbose <command>` is used
  - [ ] Traceback NOT visible in normal (non-verbose) mode
  - [ ] Error handling behavior unchanged (still ClickException with repr(e))
  - [ ] Verification: `uv run pytest tests/ -k "test_" --co -q | head` shows existing tests still collected

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: error handling still works (regression check)
    Tool: Bash (pytest)
    Steps:
      1. uv run pytest tests/cli/test_cli_checkout.py -x
      2. uv run pytest tests/remote/test_cli_remote.py -x
    Expected Result: Both pass (no regression)
  ```

  **Commit**: YES
  - Message: `fix(cli): log full traceback in verbose mode for unhandled exceptions`
  - Files: `src/pivot/cli/decorators.py`
  - Pre-commit: `uv run pytest tests/cli/test_cli_checkout.py tests/remote/test_cli_remote.py -x`

---

- [x] 3. Add tests for new behaviors

  **What to do**:
  - Add tests to `tests/cli/test_cli_checkout.py`:
    1. `test_checkout_skips_null_hash_in_stage_output` — Set up a project with a lock file containing a null hash in `output_hashes`. Run `checkout`. Assert checkout completes without error, the null-hash entry is skipped, and a warning is logged.
    2. `test_checkout_skips_empty_hash_in_stage_output` — Same but with empty string `""` hash. Assert same behavior.
    3. `test_checkout_mixed_valid_and_null_hashes` — Lock file with some valid and some null hashes. Assert valid ones are restored, null ones skipped with warning.

  - Add test to `tests/cli/test_cli_checkout.py` or a new section:
    4. `test_verbose_traceback_on_unhandled_error` — Trigger an unhandled exception in a CLI command with `--verbose` flag. Assert traceback appears in log output.

  **Must NOT do**:
  - Do not use `@pytest.mark.skip`
  - Do not create new test files (add to existing)
  - Do not mock internal functions (mock boundaries only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding tests to existing files following established patterns
  - **Skills**: [`git-master`]
    - `git-master`: Need to commit changes

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 4
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `tests/cli/test_cli_checkout.py:16-25` — `_setup_test_project()` helper — creates pivot.yaml and cache dir
  - `tests/cli/test_cli_checkout.py:28-86` — Test helper functions for setting up lock files, tracked files, cache entries
  - `tests/cli/test_cli_checkout.py:88-103` — Test structure and import patterns
  - `tests/conftest.py:397-420` — `isolated_pivot_dir` context manager

  **Test References**:
  - `tests/cli/test_cli_checkout.py:163-212` — `test_checkout_stage_output` — pattern for testing stage output checkout with lock files
  - `tests/cli/test_cli_checkout.py:757-830` — Recently added no-pipeline tests — patterns for testing without pipeline

  **Acceptance Criteria**:
  - [ ] All new tests pass: `uv run pytest tests/cli/test_cli_checkout.py -x`
  - [ ] Existing tests unchanged and still pass
  - [ ] No test uses `@pytest.mark.skip` or inline function definitions

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: All new tests pass
    Tool: Bash (pytest)
    Steps:
      1. uv run pytest tests/cli/test_cli_checkout.py tests/remote/test_cli_remote.py -x --tb=short
    Expected Result: All tests pass (exit code 0)
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `test(checkout): add tests for null-hash handling and verbose traceback`
  - Files: `tests/cli/test_cli_checkout.py`
  - Pre-commit: `uv run pytest tests/cli/test_cli_checkout.py -x`

---

- [x] 4. Run full quality checks

  **What to do**:
  - Run format, lint, type checking, and full test suite
  - Fix any issues that arise

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard quality check run
  - **Skills**: [`git-master`]
    - `git-master`: Commit any fixes

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 1, 2, 3

  **References**:
  - `AGENTS.md` — Quality commands: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
  - `AGENTS.md` — Test command: `uv run pytest tests/ -n auto`

  **Acceptance Criteria**:
  - [ ] `uv run ruff format .` → no changes needed
  - [ ] `uv run ruff check .` → no errors
  - [ ] `uv run basedpyright` → no new errors
  - [ ] `uv run pytest tests/ -n auto` → all tests pass

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Full quality checks pass
    Tool: Bash
    Steps:
      1. uv run ruff format . && uv run ruff check . && uv run basedpyright
      2. Assert: exit code 0
      3. uv run pytest tests/ -n auto
      4. Assert: exit code 0
    Expected Result: All checks green
    Evidence: Terminal output captured
  ```

  **Commit**: YES (if format changes needed)
  - Message: `chore: format`
  - Pre-commit: `uv run ruff check .`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `fix(checkout): skip null-hash entries in stage output info` | `src/pivot/cli/checkout.py` | `uv run pytest tests/cli/test_cli_checkout.py -x` |
| 2 | `fix(cli): log full traceback in verbose mode for unhandled exceptions` | `src/pivot/cli/decorators.py` | `uv run pytest tests/cli/test_cli_checkout.py tests/remote/test_cli_remote.py -x` |
| 3 | `test(checkout): add tests for null-hash handling and verbose traceback` | `tests/cli/test_cli_checkout.py` | `uv run pytest tests/cli/test_cli_checkout.py -x` |
| 4 | `chore: format` (if needed) | various | `uv run ruff format . && uv run ruff check . && uv run basedpyright` |

---

## Success Criteria

### Verification Commands
```bash
# Specific test files
uv run pytest tests/cli/test_cli_checkout.py tests/remote/test_cli_remote.py -x  # Expected: all pass

# Full suite
uv run pytest tests/ -n auto  # Expected: all pass

# Quality
uv run ruff format . && uv run ruff check . && uv run basedpyright  # Expected: clean
```

### Final Checklist
- [x] `pivot pull --all` doesn't crash on null-hash lock entries
- [x] Null-hash entries logged as warnings
- [x] `pivot --verbose pull --all` shows full tracebacks for any errors
- [x] All existing tests still pass
- [x] New tests cover null-hash scenarios
- [x] Code formatted, linted, type-checked
