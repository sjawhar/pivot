# Progress Bars for Remote Commands (push, pull, fetch, checkout)

## TL;DR

> **Quick Summary**: Replace the primitive "Uploaded X files..." echo with proper async tqdm progress bars (`tqdm.asyncio`) across all four remote commands, and add progress reporting to checkout (which currently has none). Using async tqdm ensures progress updates don't block the event loop.
> 
> **Deliverables**:
> - New `TransferProgress` class in `cli/helpers.py` wrapping `tqdm.asyncio.tqdm` with TTY/quiet detection
> - Updated callback signature in `remote/sync.py` and `remote/storage.py` to pass filename
> - Progress bars in push, pull, fetch, and checkout commands
> 
> **Estimated Effort**: Short
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (callback infrastructure) → Task 2 (push/fetch/pull) + Task 3 (checkout)

---

## Context

### Original Request
User wants progress bars on `pivot push`, `pivot pull`, `pivot fetch`, and `pivot checkout` commands.

### Interview Summary
**Key Discussions**:
- **Granularity**: Per-file (not per-byte). Matches existing callback architecture — no S3 streaming changes.
- **Library**: tqdm (specifically `tqdm.asyncio` for non-blocking async usage), consistent with existing patterns in `cli/console.py` and `cli/status.py`.
- **Display**: File count progress + current filename in bar description (e.g., `Uploading model.pkl: 42/128 [=====>] 33%`).
- **Tests**: No new automated tests — progress is UI-only, existing integration tests cover command logic.

**Research Findings**:
- Current `make_progress_callback()` in `cli/helpers.py:121-128` is extremely basic — just `click.echo(f"  {action} {completed} files...\r")` with carriage return.
- Checkout command has zero progress reporting.
- Callback signature is `Callable[[int], None]` — only knows completed count, not total or filename.
- tqdm is already used in `cli/console.py` (pipeline execution) and `cli/status.py` (file status tracking) with proper TTY detection and cleanup.
- `tqdm.asyncio` submodule provides `tqdm.asyncio.tqdm` — an async-aware subclass whose `.update()` is designed for use inside async event loops without blocking.
- `upload_batch()` and `download_batch()` in `remote/storage.py` fire the callback once per completed file, from within async tasks in the event loop.
- Total file count is known before transfer starts (after the enumerate/compare phase in `sync.py`).

### Metis Review
**Identified Gaps** (addressed):
- **Zero-file edge case**: If no files to transfer, don't create a tqdm bar at all. The existing early-return paths in `_push_async` and `_pull_async` already handle this before reaching the callback.
- **Partial failures**: tqdm bar should close properly even if transfers fail. Use try/finally pattern.
- **Output interference**: Progress bar writes to stderr; summary/error output writes to stdout. No interference.
- **Concurrent updates in checkout**: `tqdm.asyncio.tqdm` is designed for use within async event loops. Since the event loop is single-threaded, there are no race conditions on `.update()` calls.

---

## Work Objectives

### Core Objective
Add tqdm progress bars showing file count and current filename to push, pull, fetch, and checkout commands.

### Concrete Deliverables
- Updated `make_progress_callback()` in `packages/pivot/src/pivot/cli/helpers.py` → returns an async-tqdm-based callback using `tqdm.asyncio.tqdm`
- Updated callback type in `packages/pivot/src/pivot/remote/sync.py` → `Callable[[int, str], None]` (completed, filename)
- Updated callback invocation in `packages/pivot/src/pivot/remote/storage.py` → passes filename to callback
- Progress reporting added to `packages/pivot/src/pivot/cli/checkout.py` → `_checkout_files_async()`

### Definition of Done
- [ ] `pivot push` shows tqdm progress bar with file count and filename on TTY
- [ ] `pivot fetch` shows tqdm progress bar with file count and filename on TTY
- [ ] `pivot pull` shows tqdm progress bar for fetch phase AND checkout phase
- [ ] `pivot checkout` shows tqdm progress bar with file count
- [ ] `pivot push -q` suppresses progress bar
- [ ] Progress bar does not appear when output is piped (non-TTY)
- [ ] Zero files to transfer → no bar shown, no crash
- [ ] Existing tests still pass: `uv run pytest packages/pivot/tests -n auto`

### Must Have
- **Async tqdm** (`tqdm.asyncio.tqdm`) progress bar — non-blocking in async contexts
- `total`, `desc` showing current filename, file counter
- TTY detection (suppress on non-TTY)
- Quiet mode suppression (respect `--quiet` flag)
- Proper bar cleanup on success and failure (try/finally)

### Must NOT Have (Guardrails)
- No changes to S3 upload/download streaming internals
- No per-byte progress tracking
- No new CLI flags (e.g., `--progress`, `--no-progress`)
- No new dependencies
- No rich Progress bars (stick to tqdm for consistency)
- No changes to transfer logic, error handling, or return types
- No refactoring beyond progress-related code

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**

### Test Decision
- **Infrastructure exists**: YES (pytest, 90%+ coverage target)
- **Automated tests**: NO — user explicitly chose no new tests
- **Framework**: pytest (existing)
- **Agent-Executed QA**: YES (mandatory)

### Agent-Executed QA Scenarios

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| CLI progress bars | Bash (run commands with fixtures) | Run pivot commands, check exit codes, verify no crashes |
| Quiet mode | Bash (stderr capture) | Run with `-q`, assert no progress output |
| Non-TTY | Bash (pipe output) | Run piped, verify no tqdm escape codes in output |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
└── Task 1: Update callback infrastructure (helpers.py, sync.py, storage.py)

Wave 2 (After Wave 1):
├── Task 2: Wire progress bars into push/fetch/pull commands (cli/remote.py)
└── Task 3: Add progress to checkout command (cli/checkout.py)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | None |
| 2 | 1 | None | 3 |
| 3 | 1 | None | 2 |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | task(category="quick", load_skills=[], run_in_background=false) |
| 2 | 2, 3 | dispatch parallel after Wave 1 completes |

---

## TODOs

- [ ] 1. Update callback infrastructure: helpers.py, sync.py, storage.py

  **What to do**:

  1. **`cli/helpers.py`** — Replace `make_progress_callback()` with an async-tqdm-based version:
     - Import: `from tqdm.asyncio import tqdm as async_tqdm` (the async-aware subclass)
     - New signature: `make_transfer_progress(action: str, total: int, quiet: bool) -> TransferProgress`
     - `TransferProgress` should be a context manager class wrapping a `async_tqdm` bar
     - **Why async tqdm**: All progress callbacks are invoked inside async functions (`upload_one`, `download_one`, `restore_one`). Using `tqdm.asyncio.tqdm` ensures `.update()` calls don't block the event loop. Regular `tqdm.tqdm` does synchronous stderr writes which can cause micro-blocking in async contexts.
     - It should expose a callback method with signature `(completed: int, filename: str) -> None`
     - Update tqdm `desc` with current filename on each call
     - TTY detection: only create tqdm bar if `sys.stderr.isatty() and not quiet`
     - If non-TTY or quiet: callback should be a no-op (no output at all)
     - When `total == 0`: don't create a bar (early return pattern at call sites handles this)
     - Context manager `__exit__` calls `bar.close()` for cleanup
     - Remove old `make_progress_callback()` function

  2. **`remote/storage.py`** — Update callback invocation in `upload_batch()` and `download_batch()`:
     - Change callback type from `Callable[[int], None]` to `Callable[[int, str], None]`
     - In `upload_one()` (line 504): pass `local_path.name` as second arg to callback
     - In `download_one()` (line 552): pass the hash (or derive a short name) as second arg to callback
     - Keep `completed` counter as first arg for backwards compatibility of the count

  3. **`remote/sync.py`** — Update callback type annotations:
     - Change all `callback: Callable[[int], None] | None` to `Callable[[int, str], None] | None`
     - This affects: `_push_async()` (line 267), `push()` (line 328), `_pull_async()` (line 355), `pull()` (line 411)
     - No logic changes needed — sync.py just passes the callback through

  **Must NOT do**:
  - Don't change any transfer logic, error handling, or return types
  - Don't add new parameters to `push()`/`pull()` beyond the callback type change
  - Don't add total count as a parameter to sync functions — total is calculated at CLI level

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward signature changes and a focused tqdm wrapper class
  - **Skills**: `[]`
    - No special skills needed — pure Python changes
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: Not applicable — this is CLI/terminal output

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (alone)
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `packages/pivot/src/pivot/cli/console.py:97-106` — `_ensure_progress_bar()` shows the existing tqdm creation pattern: TTY check, `dynamic_ncols=True`, `leave=True`, writing to `self.stream`. Note: console.py uses regular `tqdm.tqdm` since pipeline execution is synchronous. For the async transfer/checkout context, use `tqdm.asyncio.tqdm` instead.
  - `packages/pivot/src/pivot/cli/console.py:76-80` — `close()` method shows tqdm cleanup pattern
  - `packages/pivot/src/pivot/cli/status.py:96-117` — Complete tqdm lifecycle for file tracking: TTY check, creation, `on_progress` callback updating desc/total, try/finally cleanup

  **External References** (library docs):
  - `tqdm.asyncio` module: provides `tqdm.asyncio.tqdm` — an async-aware subclass of `tqdm.tqdm`. Supports same constructor args (`total`, `desc`, `file`, `dynamic_ncols`, `leave`). Usage: `from tqdm.asyncio import tqdm as async_tqdm; bar = async_tqdm(total=N, desc="...")`

  **API/Type References** (contracts to implement against):
  - `packages/pivot/src/pivot/remote/storage.py:482-522` — `upload_batch()`: callback invoked at line 512-513, needs filename added
  - `packages/pivot/src/pivot/remote/storage.py:524-574` — `download_batch()`: callback invoked at line 564-565, needs filename added
  - `packages/pivot/src/pivot/remote/sync.py:267` — `_push_async` callback param type annotation
  - `packages/pivot/src/pivot/remote/sync.py:328` — `push` callback param type annotation
  - `packages/pivot/src/pivot/remote/sync.py:355` — `_pull_async` callback param type annotation
  - `packages/pivot/src/pivot/remote/sync.py:411` — `pull` callback param type annotation

  **Code to Remove**:
  - `packages/pivot/src/pivot/cli/helpers.py:121-128` — Old `make_progress_callback()` function

  **Acceptance Criteria**:

  - [ ] `make_progress_callback` no longer exists in `cli/helpers.py`
  - [ ] New `TransferProgress` class (or similar) exists in `cli/helpers.py` with context manager protocol, using `tqdm.asyncio.tqdm` (not regular `tqdm.tqdm`)
  - [ ] `upload_batch()` and `download_batch()` callback type is `Callable[[int, str], None] | None`
  - [ ] All type annotations in `sync.py` updated to match
  - [ ] `uv run basedpyright packages/pivot/src/pivot/cli/helpers.py packages/pivot/src/pivot/remote/sync.py packages/pivot/src/pivot/remote/storage.py` — zero new errors
  - [ ] `uv run pytest packages/pivot/tests -n auto` — all existing tests pass

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Type checking passes after callback signature change
    Tool: Bash
    Preconditions: Working pivot dev environment
    Steps:
      1. uv run basedpyright packages/pivot/src/pivot/cli/helpers.py packages/pivot/src/pivot/remote/sync.py packages/pivot/src/pivot/remote/storage.py
      2. Assert: exit code 0 (or only pre-existing errors)
    Expected Result: No new type errors introduced
    Evidence: Command output captured

  Scenario: Existing tests still pass
    Tool: Bash
    Preconditions: Working pivot dev environment
    Steps:
      1. uv run pytest packages/pivot/tests -n auto
      2. Assert: exit code 0
    Expected Result: All tests pass
    Evidence: Pytest output captured
  ```

  **Commit**: YES
  - Message: `feat(cli): add tqdm-based transfer progress bar infrastructure`
  - Files: `packages/pivot/src/pivot/cli/helpers.py`, `packages/pivot/src/pivot/remote/sync.py`, `packages/pivot/src/pivot/remote/storage.py`
  - Pre-commit: `uv run pytest packages/pivot/tests -n auto`


- [ ] 2. Wire progress bars into push, fetch, pull commands

  **What to do**:

  1. **`cli/remote.py`** — Update `push()` command (lines 92-105):
     - Before the `transfer.push()` call, calculate total: already have `local_hashes` from line 76-80. But note the actual transfer count is determined after `compare_status()` inside `_push_async()`, so we can't know the exact total at CLI level.
     - **Approach**: Pass the new `TransferProgress` callback to `transfer.push()`. Since the CLI doesn't know the exact "to-transfer" count before calling push (the compare happens inside), use `len(local_hashes)` as an upper bound OR refactor to pass total back.
     - **Simpler approach**: Have `make_transfer_progress()` accept `total=None` initially, and set the total on first callback invocation (similar to status.py pattern at line 105-111 where total is set dynamically). The `upload_batch()` already knows `len(items)` — pass that as total via a separate mechanism, OR have the callback accept total as well.
     - **Recommended approach**: Change `make_transfer_progress()` to create a bar with `total=None`. Extend the callback to `Callable[[int, int, str], None]` — `(completed, total, filename)`. The batch methods already know `len(items)`. On first call, set `bar.total`. This mirrors the `status.py` pattern exactly.
     - Wrap in context manager for cleanup: `with make_transfer_progress("Uploading", quiet) as progress:` then pass `progress.callback` to `transfer.push()`
     - Replace `cli_helpers.make_progress_callback("Uploaded")` at line 103

  2. **`cli/remote.py`** — Update `fetch()` command (lines 173-184):
     - Same pattern as push: wrap with context manager, pass callback
     - Replace `cli_helpers.make_progress_callback("Downloaded")` at line 182

  3. **`cli/remote.py`** — Update `pull()` command (lines 277-288):
     - Same pattern for the fetch phase
     - Replace `cli_helpers.make_progress_callback("Downloaded")` at line 286
     - Note: checkout phase is handled in Task 3

  **Important design decision**: The batch methods (`upload_batch`/`download_batch`) know the total file count (`len(items)`). The cleanest approach is:
  - Callback signature: `Callable[[int, int, str], None]` — `(completed, total, filename)`
  - The batch methods pass `len(items)` as `total` on every call
  - `TransferProgress.callback()` sets `bar.total` on first invocation if not set
  - This avoids needing to thread total through the sync layer

  **If Task 1 chose a different approach**, follow that — the key point is the CLI commands should use the context manager pattern for cleanup.

  **Must NOT do**:
  - Don't change the push/fetch/pull logic, error handling, or output summary format
  - Don't add new CLI flags
  - Don't change the `transfer.push()`/`transfer.pull()` function signatures beyond the callback type

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical wiring — replacing old callback creation with new context manager
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `packages/pivot/src/pivot/cli/status.py:96-117` — Dynamic total pattern: create tqdm with `total=None`, set on first callback, try/finally close
  - `packages/pivot/src/pivot/cli/remote.py:92-118` — Push command: current callback wiring at line 103
  - `packages/pivot/src/pivot/cli/remote.py:173-196` — Fetch command: current callback wiring at line 182
  - `packages/pivot/src/pivot/cli/remote.py:276-318` — Pull command: current callback wiring at line 286

  **API/Type References**:
  - Task 1 deliverables: `TransferProgress` class in `cli/helpers.py` — use its context manager and callback method

  **Acceptance Criteria**:

  - [ ] `make_progress_callback("Uploaded")` no longer appears in `cli/remote.py`
  - [ ] `make_progress_callback("Downloaded")` no longer appears in `cli/remote.py`
  - [ ] All three commands (push/fetch/pull) use context manager pattern for progress cleanup
  - [ ] `uv run basedpyright packages/pivot/src/pivot/cli/remote.py` — zero new errors
  - [ ] `uv run pytest packages/pivot/tests -n auto` — all existing tests pass

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Push command runs without error (no remote needed — just verify no crash)
    Tool: Bash
    Preconditions: Pivot project with no remote configured (or mocked)
    Steps:
      1. Run: uv run pytest packages/pivot/tests/remote/test_cli_remote.py -n auto -v
      2. Assert: exit code 0
    Expected Result: Existing remote CLI tests pass
    Evidence: Pytest output captured

  Scenario: No references to old make_progress_callback in remote.py
    Tool: Bash
    Preconditions: Task 2 complete
    Steps:
      1. grep -n "make_progress_callback" packages/pivot/src/pivot/cli/remote.py
      2. Assert: no matches found (exit code 1)
    Expected Result: Old callback factory fully replaced
    Evidence: Grep output captured

  Scenario: Type checking passes
    Tool: Bash
    Steps:
      1. uv run basedpyright packages/pivot/src/pivot/cli/remote.py
      2. Assert: exit code 0 or only pre-existing errors
    Expected Result: No new type errors
    Evidence: Command output captured
  ```

  **Commit**: YES (groups with Task 3)
  - Message: `feat(cli): add progress bars to push, fetch, pull commands`
  - Files: `packages/pivot/src/pivot/cli/remote.py`
  - Pre-commit: `uv run pytest packages/pivot/tests -n auto`


- [ ] 3. Add progress bar to checkout command

  **What to do**:

  1. **`cli/checkout.py`** — Add progress callback to `_checkout_files_async()`:
     - Add a `callback` parameter: `callback: Callable[[int, int, str], None] | None = None`
     - Inside `restore_one()` (line 127), after a successful restore (line 139: `restored += 1`), call the callback with `(restored + skipped, total, path.name)` where `total = len(files)`
     - Also after skip (line 141: `skipped += 1`), call callback similarly
     - This gives progress on every file processed, whether restored or skipped

  2. **`cli/checkout.py`** — Update `_checkout_main_async()` to accept and pass callback.

  3. **`cli/checkout.py`** — Update `checkout()` command (line 301):
     - Calculate total files: `len(tracked_files) + len(stage_outputs)` (when no targets) or `len(targets)` (when targets specified)
     - Create `TransferProgress` with `action="Restoring"` and `quiet=quiet`
     - Wrap `asyncio.run(...)` in context manager for cleanup
     - Pass callback through to `_checkout_main_async()`

  **Must NOT do**:
  - Don't change checkout logic, error handling, or file restoration behavior
  - Don't change `CheckoutBehavior` enum or `_restore_path_sync()`
  - Don't change the return type of `_checkout_files_async()`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding callback parameter threading through 3 functions
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `packages/pivot/src/pivot/cli/checkout.py:106-159` — `_checkout_files_async()`: where to add callback invocation (after line 139 and 141)
  - `packages/pivot/src/pivot/cli/checkout.py:226-256` — `_checkout_main_async()`: needs callback param passed through
  - `packages/pivot/src/pivot/cli/checkout.py:286-351` — `checkout()` command: where to create progress bar and wire it in
  - `packages/pivot/src/pivot/cli/checkout.py:127-141` — `restore_one()` inner function: the exact lines after which to invoke callback

  **API/Type References**:
  - Task 1 deliverables: `TransferProgress` class in `cli/helpers.py`
  - `packages/pivot/src/pivot/cli/checkout.py:20` — `MAX_CONCURRENT_RESTORES = 32` — concurrency is handled by semaphore, callback updates are safe (single event loop thread). The `tqdm.asyncio.tqdm` bar used by `TransferProgress` is designed for exactly this async TaskGroup pattern.

  **Acceptance Criteria**:

  - [ ] `_checkout_files_async()` accepts a callback parameter
  - [ ] `_checkout_main_async()` accepts and passes callback parameter
  - [ ] `checkout()` command creates and uses `TransferProgress` context manager
  - [ ] Progress bar shows for checkout with targets and without targets
  - [ ] `uv run basedpyright packages/pivot/src/pivot/cli/checkout.py` — zero new errors
  - [ ] `uv run pytest packages/pivot/tests/cli/test_cli_checkout.py -v` — all tests pass
  - [ ] `uv run pytest packages/pivot/tests -n auto` — all tests pass

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Checkout tests still pass
    Tool: Bash
    Steps:
      1. uv run pytest packages/pivot/tests/cli/test_cli_checkout.py -v
      2. Assert: exit code 0
    Expected Result: All checkout tests pass
    Evidence: Pytest output captured

  Scenario: Type checking passes
    Tool: Bash
    Steps:
      1. uv run basedpyright packages/pivot/src/pivot/cli/checkout.py
      2. Assert: exit code 0 or only pre-existing errors
    Expected Result: No new type errors
    Evidence: Command output captured

  Scenario: Full test suite passes
    Tool: Bash
    Steps:
      1. uv run pytest packages/pivot/tests -n auto
      2. Assert: exit code 0
    Expected Result: All tests pass
    Evidence: Pytest output captured
  ```

  **Commit**: YES (groups with Task 2)
  - Message: `feat(cli): add progress bar to checkout command`
  - Files: `packages/pivot/src/pivot/cli/checkout.py`
  - Pre-commit: `uv run pytest packages/pivot/tests -n auto`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(cli): add tqdm-based transfer progress bar infrastructure` | helpers.py, sync.py, storage.py | `uv run pytest packages/pivot/tests -n auto` |
| 2+3 | `feat(cli): add progress bars to push, fetch, pull, checkout commands` | remote.py, checkout.py | `uv run pytest packages/pivot/tests -n auto` |

---

## Success Criteria

### Verification Commands
```bash
uv run pytest packages/pivot/tests -n auto          # All tests pass
uv run basedpyright                                   # No new type errors
uv run ruff format . && uv run ruff check .           # Code quality
```

### Final Checklist
- [ ] All four commands (push, pull, fetch, checkout) show tqdm progress bars on TTY
- [ ] Progress bars use `tqdm.asyncio.tqdm` (non-blocking in async contexts)
- [ ] Progress bars show filename and file count
- [ ] Progress suppressed with `--quiet` and on non-TTY
- [ ] No new dependencies added
- [ ] No S3 streaming changes
- [ ] All existing tests pass
- [ ] Type checking passes
