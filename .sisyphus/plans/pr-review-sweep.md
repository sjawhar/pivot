# PR Review Sweep: Make It Shine

## TL;DR

> **Quick Summary**: Fix all issues found by 24 parallel review agents across 12 PRs merged Feb 7-8. Focus on dead code removal, unaddressed Copilot comments, incomplete migrations, documentation accuracy, and missing test coverage.
>
> **Deliverables**:
> - Dead code and migration leftovers removed
> - All still-applicable Copilot comments addressed
> - Path canonicalization migration completed
> - Stale plan docs cleaned up
> - Missing error handling added
> - Test gaps filled
>
> **Estimated Effort**: Medium (many small, well-scoped fixes)
> **Parallel Execution**: YES — 7 waves
> **Critical Path**: Task 1 (path canonicalization) → Task 9 (run quality checks)

---

## Context

### Original Request
Review all 12 PRs merged on Feb 7-8 for issues that slipped through review, especially unaddressed Copilot comments. Then produce a fix plan covering dead code, migration leftovers, duplication, design issues, and general polish.

### Research Summary
24 agents dispatched in parallel (12 independent reviewers + 12 Copilot comment checkers) across PRs #386–#398. Key findings:

- **Copilot comments**: 31 total across all PRs. 13 already fixed before merge. **18 still apply** to current codebase.
- **Independent reviews**: Found additional dead code, incomplete migrations, design issues, and test gaps.
- **PRs with zero issues**: #390 (exemplary code), #391 (minor only)
- **PRs with most issues**: #392 (incomplete path migration), #396 (misleading docs), #398 (missing error handling)
- **Cross-cutting**: Plan docs committed with wrong architecture descriptions and AI assistant directives

---

## Work Objectives

### Core Objective
Address all actionable findings from the review sweep — eliminate dead code, fix unaddressed Copilot feedback, complete incomplete migrations, and improve test coverage.

### Concrete Deliverables
- Clean code with no dead functions, unused fields, or migration leftovers
- Accurate documentation and comments
- Consistent error handling patterns
- Complete path canonicalization migration
- Filled test gaps

### Definition of Done
- [x] `uv run ruff format . && uv run ruff check . && uv run basedpyright` clean
- [x] `uv run pytest tests/ -n auto` passes (90%+ coverage maintained)
- [x] No remaining dead code from removed features
- [x] All Copilot comments that still apply are addressed

### Must Have
- Fix all bug-risk items (missing error handling, incorrect accounting)
- Remove all dead code and migration leftovers
- Fix all inaccurate documentation/comments
- Complete path canonicalization migration in worker.py, checkout.py, verify.py

### Must NOT Have (Guardrails)
- No new features — only fixes and cleanup
- No architecture changes — keep existing patterns, just complete/fix them
- No changes to public API behavior
- No unnecessary abstractions — inline fixes over new helper classes
- Do NOT touch PR #390 code (it was rated exemplary)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: Tests-after (add missing tests, verify existing pass)
- **Framework**: pytest with pytest-xdist

### Agent-Executed QA Scenarios

Every task ends with:
```
uv run ruff format . && uv run ruff check . && uv run basedpyright
uv run pytest tests/ -n auto
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — independent fixes):
├── Task 1: Complete path canonicalization migration (PR #392)
├── Task 2: Fix missing ImportError handling in repro.py (PR #398)
├── Task 3: Fix TransferSummary accounting + KeyError handling (PR #389, #388)
└── Task 4: Remove dead code and unused fields (PRs #386, #396, #398)

Wave 2 (After Wave 1):
├── Task 5: Fix documentation/comment accuracy (PRs #390, #392, #396)
└── Task 6: Fix GraphView non-determinism + sort stages/artifacts (PR #395)

Wave 3 (After Wave 2):
├── Task 7: Clean up plan docs (PRs #390, #391, #394, #396)
└── Task 8: Fill test gaps and fix existing tests (PRs #389, #394, #397)

Wave 4 (Final):
└── Task 9: Run full quality checks and fix any breakage
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|-----------|--------|---------------------|
| 1 | None | 5, 9 | 2, 3, 4 |
| 2 | None | 9 | 1, 3, 4 |
| 3 | None | 9 | 1, 2, 4 |
| 4 | None | 9 | 1, 2, 3 |
| 5 | 1 | 9 | 6 |
| 6 | None | 9 | 5 |
| 7 | None | 9 | 8 |
| 8 | None | 9 | 7 |
| 9 | 1-8 | None | None (final) |

---

## TODOs

- [x] 1. Complete path canonicalization migration

  **What to do**:
  - Replace `project.normalize_path() + preserve_trailing_slash()` with `path_utils.canonicalize_artifact_path()` in:
    - `src/pivot/executor/worker.py` — `normalize_out_path()` function and lines ~955, 1032, 1061
    - `src/pivot/cli/checkout.py:48-50, 61-62`
    - `src/pivot/cli/verify.py:83-85, 93`
  - Fix `canonicalize_artifact_path()` itself: use `normalized.as_posix()` instead of `str(normalized)` at `src/pivot/path_utils.py:35` (Copilot PR #392 comment — POSIX separator promise)
  - Fix backslash detection at `src/pivot/path_utils.py:29-31` — only check forward slash since output is always POSIX
  - Fix project root escape check at `src/pivot/pipeline/pipeline.py:267-273` — apply to ALL paths, not just relative ones (absolute paths like `/tmp/evil.txt` bypass validation)
  - Consider removing `preserve_trailing_slash()` as standalone function (its logic is in `canonicalize_artifact_path` now) — or mark with deprecation comment
  - Pass `state_db` to `worker.hash_output()` call in `src/pivot/executor/commit.py:152` for hash caching performance (NOTE: explain.py was originally flagged but does not contain hash_output calls — verified by Momus)

  **Must NOT do**:
  - Don't change the canonical path format itself
  - Don't touch lock.py boundary conversion (it correctly converts absolute↔relative)
  - Don't refactor normalize_path for non-artifact paths

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`analyze`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 5, Task 9
  - **Blocked By**: None

  **References**:
  - `src/pivot/path_utils.py:9-49` — canonicalize_artifact_path and preserve_trailing_slash definitions
  - `src/pivot/executor/worker.py:402-405` — normalize_out_path function to replace
  - `src/pivot/cli/checkout.py:48-62` — old normalization pattern to migrate
  - `src/pivot/cli/verify.py:83-93` — old normalization pattern to migrate
  - `src/pivot/pipeline/pipeline.py:263-273` — path resolution and escape check
  - `src/pivot/executor/commit.py:152` — missing state_db param for hash caching

  **Acceptance Criteria**:
  - [ ] `grep -r "preserve_trailing_slash" src/pivot/` shows only path_utils.py definition (or none)
  - [ ] `grep -r "normalize_out_path" src/pivot/` shows zero results (function removed)
  - [ ] `path_utils.canonicalize_artifact_path()` uses `.as_posix()` for POSIX separators
  - [ ] Escape check applies to absolute paths too
  - [ ] `uv run pytest tests/ -n auto` passes
  - [ ] `uv run basedpyright` clean

  **Commit**: YES
  - Message: `refactor(paths): complete canonicalize_artifact_path migration`
  - Files: `src/pivot/path_utils.py`, `src/pivot/executor/worker.py`, `src/pivot/cli/checkout.py`, `src/pivot/cli/verify.py`, `src/pivot/pipeline/pipeline.py`, `src/pivot/executor/commit.py`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 2. Fix missing ImportError handling in repro.py

  **What to do**:
  - Wrap bare `import pivot_tui.run as tui_run` at `src/pivot/cli/repro.py:354` and `src/pivot/cli/repro.py:585` with try/except ImportError → click.UsageError
  - Match the pattern already used in `src/pivot/cli/run.py` and `src/pivot/cli/data.py`
  - Standardize error messages across all 4 locations (repro.py x2, run.py, data.py) to use consistent wording: pick either "pip install 'pivot[tui]'" or "uv pip install pivot-tui", not both
  - Add explanatory comment to the bare `except OSError: pass` in `packages/pivot-tui/tests/helpers.py:38-39` (minor Copilot nit from PR #398)

  **Must NOT do**:
  - Don't change the lazy import pattern itself (it's correct)
  - Don't add eager imports of pivot_tui

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`analyze`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Task 9
  - **Blocked By**: None

  **References**:
  - `src/pivot/cli/run.py:83-88` — existing try/except ImportError pattern to follow
  - `src/pivot/cli/data.py:128-133` — existing try/except ImportError pattern to follow
  - `src/pivot/cli/repro.py:354,585` — locations to fix
  - `packages/pivot-tui/tests/helpers.py:38-39` — OSError pass to comment

  **Acceptance Criteria**:
  - [ ] Both `import pivot_tui.run` in repro.py wrapped in try/except
  - [ ] Error messages consistent across run.py, data.py, repro.py
  - [ ] `uv run pytest tests/ -n auto` passes

  **Commit**: YES
  - Message: `fix(cli): add ImportError handling for pivot-tui imports in repro.py`
  - Files: `src/pivot/cli/repro.py`, `src/pivot/cli/run.py`, `src/pivot/cli/data.py`, `packages/pivot-tui/tests/helpers.py`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 3. Fix TransferSummary accounting + KeyError handling in sync

  **What to do**:
  - `src/pivot/remote/sync.py:290-313` — add `skipped_non_file` to the `skipped` count in the returned TransferSummary: `skipped=len(status["common"]) + skipped_non_file`
  - `src/pivot/remote/sync.py:63-71` (`get_stage_output_hashes`) — wrap `cli_helpers.get_stage(stage_name)` in try/except KeyError with logger.warning and continue
  - `packages/pivot-tui/src/pivot_tui/run.py:1106-1129` (`action_commit`) — add `_commit_in_progress` / `_cancel_commit` guards (flags already declared at line 256-257 but never used)

  **Must NOT do**:
  - Don't change TransferSummary type definition
  - Don't change the push/pull flow logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`analyze`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 9
  - **Blocked By**: None

  **References**:
  - `src/pivot/remote/sync.py:289-313` — skipped_non_file counting and TransferSummary return
  - `src/pivot/remote/sync.py:63-71` — get_stage_output_hashes with unguarded KeyError
  - `packages/pivot-tui/src/pivot_tui/run.py:256-257` — _commit_in_progress/_cancel_commit flag declarations
  - `packages/pivot-tui/src/pivot_tui/run.py:935-936` — action_escape_action checks _commit_in_progress
  - `packages/pivot-tui/src/pivot_tui/run.py:1106-1129` — action_commit implementation to fix

  **Acceptance Criteria**:
  - [ ] TransferSummary.skipped includes skipped_non_file count
  - [ ] get_stage_output_hashes handles KeyError gracefully with warning
  - [ ] action_commit checks/sets _commit_in_progress, honors _cancel_commit, has finally cleanup
  - [ ] `uv run pytest tests/ -n auto` passes

  **Commit**: YES
  - Message: `fix(remote,tui): fix TransferSummary accounting, KeyError handling, commit guards`
  - Files: `src/pivot/remote/sync.py`, `packages/pivot-tui/src/pivot_tui/run.py`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 4. Remove dead code, unused fields, and migration leftovers

  **What to do**:
  - `src/pivot/storage/lock.py:234-262` — delete `get_pending_stages_dir()`, `get_pending_lock()`, `list_pending_stages()` (pending lock system removed in PR #386, functions are dead)
  - `src/pivot/executor/worker.py:805` — remove `self._max_lines = max_lines` (unused, deque maxlen is sufficient)
  - `src/pivot/executor/worker.py:867-877` — remove dead `if self._read_fd is None: break` check in `_pipe_reader()` (read_fd is never None during execution)
  - `src/pivot/engine/engine.py:725-734` — fix duplicate sentinel sends: remove the happy-path send and let the finally block handle it (it runs on ALL paths anyway)
  - `.sisyphus/boulder.json`, `.sisyphus/drafts/tui-package-extraction.md`, `.sisyphus/notepads/tui-package-extraction/` — remove committed session files, add `.sisyphus/` to `.gitignore` (except `.sisyphus/plans/` if desired)
  - Check if `normalize_out_path` in worker.py was already removed by Task 1; if not, remove it here

  **Must NOT do**:
  - Don't remove `preserve_trailing_slash()` here (handled in Task 1)
  - Don't remove functions that are still called (verify with grep first)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`analyze`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 9
  - **Blocked By**: None

  **References**:
  - `src/pivot/storage/lock.py:234-262` — dead pending lock functions
  - `src/pivot/executor/worker.py:805` — unused _max_lines field
  - `src/pivot/executor/worker.py:867-877` — dead None check in _pipe_reader
  - `src/pivot/engine/engine.py:725-734` — duplicate sentinel logic
  - `.sisyphus/boulder.json` — session file to remove
  - `.gitignore` — add .sisyphus/ exclusion

  **Acceptance Criteria**:
  - [ ] `grep -r "get_pending_stages_dir\|get_pending_lock\|list_pending_stages" src/` — zero results
  - [ ] `grep -r "_max_lines" src/pivot/executor/worker.py` — zero results
  - [ ] `.sisyphus/boulder.json` removed
  - [ ] `.gitignore` includes `.sisyphus/` pattern
  - [ ] `uv run pytest tests/ -n auto` passes

  **Commit**: YES
  - Message: `chore: remove dead code, unused fields, and session artifacts`
  - Files: `src/pivot/storage/lock.py`, `src/pivot/executor/worker.py`, `src/pivot/engine/engine.py`, `.gitignore`, `.sisyphus/*`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 5. Fix documentation and comment accuracy

  **What to do**:
  - `src/pivot/registry.py:65-66` — change "the one boundary" to "a key boundary" and mention output index cache as another
  - `src/pivot/pipeline/pipeline.py:232` — update docstring to mention output index as another conversion boundary
  - `src/pivot/types.py:207-209` — change "ONLY place" to "only place in lockfiles" and scope the claim correctly
  - `src/pivot/types.py:131` — change "absent → skip" to "False/absent → skip"
  - `src/pivot/engine/engine.py:546-550` — fix misleading comment about Manager (it says Manager was replaced, but Manager is still used)
  - `src/pivot/cli/_run_common.py:133,153` — update error messages to mention `--jsonl/--json` (since --json is an alias)
  - `tests/remote/test_transfer.py:146,226` — update two outdated test docstrings (say "including manifest" but tree hash is now excluded)
  - `tests/conftest.py:77-78` — add comment explaining why OSError/ValueError is silently caught in cgroup detection
  - `tests/conftest.py` — add comments for magic numbers: 2GB per worker, 16 cap, 8 fallback, 2^60 threshold
  - `src/pivot/executor/commit.py` — add docstring note that commits are not atomic across stages

  **Must NOT do**:
  - Don't change behavior — only comments, docstrings, and error message text
  - Don't refactor anything in this task

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`analyze`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 6)
  - **Blocks**: Task 9
  - **Blocked By**: Task 1 (types.py and registry.py may have path-related changes)

  **References**:
  - `src/pivot/registry.py:65-66` — "one boundary" claim
  - `src/pivot/pipeline/pipeline.py:232` — docstring mentioning only lockfiles
  - `src/pivot/types.py:207-209` — "ONLY place" claim
  - `src/pivot/types.py:131` — increment_outputs comment
  - `src/pivot/engine/engine.py:546-550` — misleading Manager comment
  - `src/pivot/cli/_run_common.py:133,153` — error messages
  - `tests/remote/test_transfer.py:146,226` — outdated docstrings
  - `tests/conftest.py:77-78` — bare except pass
  - `src/pivot/executor/commit.py:30-45` — commit_stages docstring

  **Acceptance Criteria**:
  - [ ] No comment claims lockfiles are the "one" or "ONLY" boundary for relative paths
  - [ ] Error messages mention both --jsonl and --json
  - [ ] All magic numbers in conftest.py have explanatory comments
  - [ ] `uv run basedpyright` clean

  **Commit**: YES
  - Message: `docs: fix inaccurate comments, docstrings, and error messages`
  - Files: multiple (see list above)
  - Pre-commit: `uv run basedpyright`

---

- [x] 6. Fix GraphView non-determinism

  **What to do**:
  - `src/pivot/engine/graph.py:511-516` — sort `stages` and `artifacts` lists in the GraphView return (edges already sorted, but node lists are not)
  - This ensures deterministic output from ASCII, Mermaid, and DOT renderers

  **Must NOT do**:
  - Don't change GraphView type definition
  - Don't change the graph building logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`analyze`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 5)
  - **Blocks**: Task 9
  - **Blocked By**: None

  **References**:
  - `src/pivot/engine/graph.py:466-516` — extract_graph_view function, especially the return statement
  - `src/pivot/dag/render.py` — renderers that consume GraphView (depend on deterministic ordering)

  **Acceptance Criteria**:
  - [ ] `stages=sorted(stages)` and `artifacts=sorted(artifacts)` in the return
  - [ ] `uv run pytest tests/ -n auto` passes (renderer tests should still pass with sorted order)

  **Commit**: YES (groups with Task 5)
  - Message: `fix(graph): sort GraphView stages/artifacts for deterministic rendering`
  - Files: `src/pivot/engine/graph.py`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 7. Clean up stale plan docs

  **What to do**:
  - Remove or fix plan documents that have inaccurate content:
    - `docs/plans/2026-02-07-cli-consistency-failure-defaults-jsonl.md` — completed plan, 562 lines, remove entirely
    - `docs/plans/2026-02-07-split-deferred-writes.md:338,431` — fix code snippet and summary table to match actual implementation (`in (StageStatus.RAN, StageStatus.SKIPPED)` not `!= StageStatus.FAILED`)
    - `docs/plans/2026-02-07-logging-transport-replace-manager-queue.md` — fix architecture description (still uses Manager, not spawn_ctx.Queue)
    - `docs/plans/2026-02-07-worker-output-capture-bound-memory-fd.md:3` — remove AI assistant directive ("For Claude: REQUIRED SUB-SKILL...")
    - `docs/plans/2026-02-07-worker-output-capture-bound-memory-fd.md:31` — fix pipe lifecycle description (lazy init via _ensure_pipe, not __enter__)
    - `docs/plans/2026-02-07-watch-mode-restart-workers.md:15,45,106` — remove references to deleted `no_cache` field
    - `docs/plans/2026-02-07-canonical-artifact-paths.md:502` — update to reflect migration completion (from Task 1)
  - Remove AI assistant directives from ALL plan docs (grep for "For Claude", "SUB-SKILL", "superpowers")

  **Must NOT do**:
  - Don't delete plan docs that are still accurate and useful as historical reference
  - Don't rewrite plans — just fix factual errors and remove noise

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`analyze`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 8)
  - **Blocks**: Task 9
  - **Blocked By**: None

  **References**:
  - `docs/plans/` — all plan documents from Feb 7
  - `src/pivot/engine/engine.py:546-550,611-613` — actual implementation to verify against plans

  **Acceptance Criteria**:
  - [ ] `grep -r "For Claude\|SUB-SKILL\|superpowers" docs/plans/` — zero results
  - [ ] `grep -r "no_cache" docs/plans/2026-02-07-watch-mode` — zero results
  - [ ] Completed CLI plan doc removed
  - [ ] All remaining plan docs accurately describe what was actually implemented

  **Commit**: YES
  - Message: `docs: fix inaccurate plan docs and remove AI directives`
  - Files: `docs/plans/*.md`
  - Pre-commit: none needed (docs only)

---

- [x] 8. Fill test gaps and fix existing tests

  **What to do**:
  - `tests/remote/test_sync.py:155-185` — rewrite `test_push_skips_directory_cache_paths` to actually call `sync.push()` or `_push_async()` with a mock remote, verifying `upload_batch` only receives file paths (currently re-implements filtering logic inline instead of testing the real function)
  - `tests/cli/test_run.py:194-209` and `tests/cli/test_cli_run_keep_going.py:288-305` — add assertions verifying the second stage was NOT executed in fail-fast tests (currently only check failing stage shows "FAILED")
  - `tests/cli/test_cli_run_keep_going.py:307-332` — either make `test_run_fail_fast_stops_early` test something meaningful or remove it (currently just verifies flag is accepted, same as help test)
  - `tests/cli/test_cli_run_common.py:271-272` — rename `test_validate_tui_log_raises_for_json` to `test_validate_tui_log_raises_for_jsonl` and update docstring (PR #394 Copilot comment)
  - `tests/remote/test_sync.py` and `tests/remote/test_transfer.py` — deduplicate `_mock_get_stage` autouse fixture into `tests/remote/conftest.py`

  **Must NOT do**:
  - Don't add tests for `_get_cgroup_memory_limit_bytes` (it reads system files that vary per environment — test value is low)
  - Don't rewrite working tests — only fix the specific gaps identified
  - Don't change test infrastructure

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`analyze`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: None

  **References**:
  - `tests/remote/test_sync.py:155-185` — push directory filtering test to rewrite
  - `tests/cli/test_run.py:194-209` — fail-fast test missing assertion
  - `tests/cli/test_cli_run_keep_going.py:288-332` — fail-fast tests to fix
  - `tests/cli/test_cli_run_common.py:271-272` — test to rename
  - `tests/remote/conftest.py` — target for deduplicated fixture

  **Acceptance Criteria**:
  - [ ] Push test calls actual sync function, not reimplemented logic
  - [ ] Fail-fast tests verify second stage did NOT execute
  - [ ] Renamed test matches actual behavior being tested
  - [ ] `uv run pytest tests/ -n auto` passes with all new/modified tests

  **Commit**: YES
  - Message: `test: fill test gaps for push filtering, fail-fast behavior, fixture dedup`
  - Files: `tests/remote/test_sync.py`, `tests/cli/test_run.py`, `tests/cli/test_cli_run_keep_going.py`, `tests/cli/test_cli_run_common.py`, `tests/remote/conftest.py`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 9. Run full quality checks and fix any breakage

  **What to do**:
  - Run `uv run ruff format .`
  - Run `uv run ruff check .` — fix any issues
  - Run `uv run basedpyright` — fix any type errors introduced
  - Run `uv run pytest tests/ -n auto` — fix any test failures
  - Verify no regressions from all previous tasks

  **Must NOT do**:
  - Don't introduce new functionality
  - Don't suppress warnings with blanket ignores

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`test-and-fix`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final, sequential)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 1-8

  **References**:
  - All files modified in Tasks 1-8
  - `pyproject.toml` — ruff and basedpyright configuration

  **Acceptance Criteria**:
  - [ ] `uv run ruff format . && uv run ruff check .` — zero issues
  - [ ] `uv run basedpyright` — zero errors
  - [ ] `uv run pytest tests/ -n auto` — all tests pass, 90%+ coverage

  **Commit**: YES
  - Message: `chore: fix lint, type-check, and test issues from review sweep`
  - Files: any files needing fixup
  - Pre-commit: full suite

---

## Commit Strategy

| After Task | Message | Verification |
|------------|---------|-------------|
| 1 | `refactor(paths): complete canonicalize_artifact_path migration` | pytest |
| 2 | `fix(cli): add ImportError handling for pivot-tui imports in repro.py` | pytest |
| 3 | `fix(remote,tui): fix TransferSummary accounting, KeyError handling, commit guards` | pytest |
| 4 | `chore: remove dead code, unused fields, and session artifacts` | pytest |
| 5 | `docs: fix inaccurate comments, docstrings, and error messages` | basedpyright |
| 6 | `fix(graph): sort GraphView stages/artifacts for deterministic rendering` | pytest |
| 7 | `docs: fix inaccurate plan docs and remove AI directives` | none |
| 8 | `test: fill test gaps for push filtering, fail-fast behavior, fixture dedup` | pytest |
| 9 | `chore: fix lint, type-check, and test issues from review sweep` | full suite |

---

## Success Criteria

### Verification Commands
```bash
uv run ruff format . && uv run ruff check .        # Expected: clean
uv run basedpyright                                  # Expected: zero errors
uv run pytest tests/ -n auto                         # Expected: all pass, 90%+ coverage
grep -r "preserve_trailing_slash" src/pivot/          # Expected: only path_utils.py (or none)
grep -r "normalize_out_path" src/pivot/               # Expected: zero
grep -r "get_pending_stages_dir" src/pivot/           # Expected: zero
grep -r "For Claude\|SUB-SKILL" docs/plans/           # Expected: zero
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] All Copilot comments addressed
- [x] All dead code removed
- [x] Path canonicalization migration complete
- [x] Plan docs accurate
- [x] Tests pass with coverage maintained
