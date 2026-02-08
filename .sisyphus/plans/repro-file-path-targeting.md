# Resolve File Paths to Stages in `pivot repro`

## TL;DR

> **Quick Summary**: Make `pivot repro` accept output file paths as arguments (not just stage names), resolving them to the stages that produce those files. This reuses the exact pattern already working in `pivot dag`.
> 
> **Deliverables**:
> - Shared `resolve_targets_to_stages()` function in `cli/targets.py`
> - Updated `repro` command to resolve file paths before stage validation
> - Updated error messages for unresolved targets
> - Tests for the new resolution logic
> 
> **Estimated Effort**: Short
> **Parallel Execution**: NO — sequential (3 tasks, each builds on the last)
> **Critical Path**: Task 1 → Task 2 → Task 3

---

## Context

### Original Request
User runs `pivot repro --force ../../data/base/raw/agent_runs/*.jsonl` and gets:
```
Error: Unknown stage(s): ../../data/base/raw/agent_runs/mp4-server.jsonl, ...
```
Because `repro` only accepts stage names, not output file paths. The `pivot dag` command already supports this exact feature.

### Interview Summary
**Key Discussions**:
- Pipeline discovery works fine — the issue is purely argument resolution in `repro`
- `pivot dag` already resolves artifact paths → producer stages via `_resolve_targets_to_stages()`
- Resolution strategy: same as `dag` — stage name first, fall back to artifact path lookup
- Scope: just `repro`, not `run`
- No new CLI flags needed

**Research Findings**:
- `cli/dag.py:_resolve_targets_to_stages()` (lines 18-47) does the exact logic needed
- `engine/graph.py:get_producer()` (line 292) does O(1) lookup of artifact path → producer stage via bipartite graph
- `cli/targets.py:_classify_targets()` handles disambiguation (stage name wins over file path)
- The change point in `repro.py` is lines 844-845: `stages_to_list()` then `validate_stages_exist()`
- Shell expands globs before Click sees them, so individual file paths arrive as separate arguments

### Metis Review
**Identified Gaps** (addressed):
- Path normalization mismatch: mitigated by using `project.normalize_path()` consistently (same as `dag.py`)
- Duplicate stage resolution: if two paths resolve to same stage, dedup in the set
- Directory targets: out of scope — only file paths and stage names
- Paths outside project root: will naturally fail to match any registered output (produces helpful error)

---

## Work Objectives

### Core Objective
When `pivot repro` receives arguments that aren't stage names, resolve them as output file paths to their producer stages — identical to how `pivot dag` already handles this.

### Concrete Deliverables
- `cli/targets.py`: new public `resolve_targets_to_stages()` function (extracted from `dag.py`)
- `cli/dag.py`: refactored to call the shared function
- `cli/repro.py`: use `resolve_targets_to_stages()` instead of raw `validate_stages_exist()`
- `tests/cli/test_cli_targets.py`: unit tests for the new function
- `tests/cli/test_repro.py`: integration tests for file-path repro

### Definition of Done
- [x] `pivot repro output.csv` resolves to the stage that produces `output.csv` and runs it with deps
- [x] `pivot repro stage_name` still works unchanged
- [x] `pivot repro ../../relative/path.csv` resolves correctly via path normalization
- [x] `pivot repro nonexistent.csv` produces a clear error referencing both stages and output paths
- [x] `uv run pytest packages/pivot/tests -n auto` passes
- [x] `uv run ruff format . && uv run ruff check . && uv run basedpyright` passes

### Must Have
- Stage name takes precedence over file path when ambiguous (existing convention)
- Path normalization uses `project.normalize_path()` (existing convention)
- Works with relative paths (e.g., `../../data/file.csv`) from any subdirectory
- Deduplicates when multiple paths resolve to the same stage

### Must NOT Have (Guardrails)
- Do NOT add new CLI flags or options
- Do NOT modify `pivot run`
- Do NOT change pipeline discovery or registry semantics
- Do NOT add directory/glob resolution (shell handles globs; directories are out of scope)
- Do NOT add interactive disambiguation prompts (keep scriptable)
- Do NOT modify `pivot dag` behavior (only refactor to share code)

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**

### Test Decision
- **Infrastructure exists**: YES (pytest, 90%+ coverage)
- **Automated tests**: YES (tests-after)
- **Framework**: pytest via `uv run pytest`

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Shared function** | Bash (pytest) | Run unit tests, assert pass |
| **CLI integration** | Bash (pytest + Click CliRunner) | Run integration tests, assert pass |
| **Code quality** | Bash (ruff + basedpyright) | Run quality checks, assert zero errors |

---

## Execution Strategy

### Sequential Execution

```
Task 1: Extract resolve_targets_to_stages to shared targets.py
    ↓
Task 2: Wire into repro.py (replace validate_stages_exist flow)
    ↓
Task 3: Add tests for new behavior
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1 | None | 2, 3 |
| 2 | 1 | 3 |
| 3 | 1, 2 | None |

### Agent Dispatch Summary

| Task | Recommended Dispatch |
|------|---------------------|
| 1 | task(category="quick", load_skills=[], ...) |
| 2 | task(category="quick", load_skills=[], ...) |
| 3 | task(category="quick", load_skills=[], ...) |

---

## TODOs

- [x] 1. Extract `resolve_targets_to_stages()` into shared `cli/targets.py`

  **What to do**:
  - Add a new public function `resolve_targets_to_stages()` to `packages/pivot/src/pivot/cli/targets.py`
  - The function takes a list of target strings and returns resolved stage names + unresolved targets
  - Logic: for each target, check if it's a registered stage name (exact match). If not, normalize the path via `project.normalize_path()`, then use `engine_graph.get_producer(bipartite_graph, norm_path)` to find the producer stage. Collect unresolved targets.
  - Refactor `cli/dag.py:_resolve_targets_to_stages()` to call the new shared function instead of duplicating logic
  - The function needs the bipartite graph as input (caller builds it). Follow the same signature pattern as `dag.py`'s version.

  **Must NOT do**:
  - Do NOT change the behavior of `pivot dag` — only refactor to share code
  - Do NOT add filesystem existence checks (unlike `_classify_targets`, this resolves via the DAG, not the filesystem)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
    - No special skills needed — straightforward Python refactoring

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (first task)
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `packages/pivot/src/pivot/cli/dag.py:18-47` — `_resolve_targets_to_stages()`: the exact function to extract. Note how it iterates targets, checks `registered_stages`, then falls back to `engine_graph.get_producer(bipartite_graph, norm_path)`
  - `packages/pivot/src/pivot/cli/targets.py:49-77` — `_classify_targets()`: existing target resolution in targets.py showing the module's conventions (TypedDict results, project root handling)

  **API/Type References**:
  - `packages/pivot/src/pivot/engine/graph.py:292-308` — `get_producer(g, path)`: takes bipartite graph + Path, returns stage name or None
  - `packages/pivot/src/pivot/engine/graph.py:73-87` — `_build_outputs_map()`: shows the output path → stage mapping pattern
  - `packages/pivot/src/pivot/engine/graph.py:57-61` — `stage_node()`, `artifact_node()`: node naming convention
  - `packages/pivot/src/pivot/project.py` — `normalize_path()`: path normalization used by dag.py

  **Test References**:
  - `packages/pivot/tests/cli/test_dag_cmd.py:362-382` — `test_dag_target_artifact_path`: integration test showing artifact path resolution working end-to-end in dag command
  - `packages/pivot/tests/cli/test_cli_targets.py` — existing target resolution tests (structure and fixture patterns to follow)

  **Acceptance Criteria**:

  - [x] `resolve_targets_to_stages()` is a public function in `cli/targets.py`
  - [x] It accepts targets list and bipartite graph, returns `(set[str], list[str])` — (resolved stage names, unresolved targets)
  - [x] `cli/dag.py:_resolve_targets_to_stages()` now delegates to the shared function (or is removed if dag.py can call targets directly)
  - [x] `pivot dag clean.csv --stages` still works (no behavior change)
  - [x] `uv run ruff format . && uv run ruff check . && uv run basedpyright` passes

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: dag command still works with artifact paths after refactor
    Tool: Bash (pytest)
    Preconditions: None
    Steps:
      1. uv run pytest packages/pivot/tests/cli/test_dag_cmd.py -x -q
      2. Assert: exit code 0, all tests pass
    Expected Result: All existing dag tests pass unchanged
    Evidence: pytest output captured

  Scenario: Code quality passes
    Tool: Bash
    Preconditions: None
    Steps:
      1. uv run ruff format --check packages/pivot/src/pivot/cli/targets.py packages/pivot/src/pivot/cli/dag.py
      2. uv run ruff check packages/pivot/src/pivot/cli/targets.py packages/pivot/src/pivot/cli/dag.py
      3. uv run basedpyright packages/pivot/src/pivot/cli/targets.py packages/pivot/src/pivot/cli/dag.py
      4. Assert: all exit code 0
    Expected Result: Zero formatting, lint, or type errors
    Evidence: Command output captured
  ```

  **Commit**: YES
  - Message: `refactor(cli): extract resolve_targets_to_stages to shared targets module`
  - Files: `packages/pivot/src/pivot/cli/targets.py`, `packages/pivot/src/pivot/cli/dag.py`
  - Pre-commit: `uv run pytest packages/pivot/tests/cli/test_dag_cmd.py -x -q`

---

- [x] 2. Wire `resolve_targets_to_stages()` into `repro` command

  **What to do**:
  - In `packages/pivot/src/pivot/cli/repro.py`, replace the current flow at lines 844-845:
    ```python
    stages_list = cli_helpers.stages_to_list(stages)
    cli_helpers.validate_stages_exist(stages_list)
    ```
    With a flow that:
    1. Converts args to list via `cli_helpers.stages_to_list(stages)`
    2. If stages_list is not None, builds the bipartite graph via `engine_graph.build_graph(cli_helpers.get_all_stages())`
    3. Calls `cli_targets.resolve_targets_to_stages(stages_list, bipartite_graph)` to resolve file paths to stages
    4. If there are unresolved targets, raises a clear error (similar to `StageNotFoundError` but mentioning both stages and output paths)
    5. Passes the resolved stage names to the rest of the function
  - Update the `repro` docstring to document that STAGES can be stage names or output file paths
  - Update the shell completion (if needed) — currently `complete_stages` only completes stage names. Consider if file path completion is wanted (likely not needed — shell handles file completion natively).

  **Must NOT do**:
  - Do NOT add new CLI flags
  - Do NOT change behavior when no stages are passed (still runs entire pipeline)
  - Do NOT change `--dry-run`, `--explain`, or `--force` semantics

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
    - Simple CLI wiring — no special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `packages/pivot/src/pivot/cli/dag.py:104-124` — How `dag_cmd` builds the bipartite graph and calls target resolution (lines 105-111 specifically)
  - `packages/pivot/src/pivot/cli/repro.py:844-845` — The exact lines to replace (stages_to_list + validate_stages_exist)
  - `packages/pivot/src/pivot/cli/repro.py:828-834` — The repro docstring to update

  **API/Type References**:
  - `packages/pivot/src/pivot/cli/helpers.py:66-69` — `get_all_stages()`: returns dict needed for `build_graph()`
  - `packages/pivot/src/pivot/cli/helpers.py:72-74` — `build_dag()`: alternative that builds from pipeline context
  - `packages/pivot/src/pivot/engine/graph.py:135-200` — `build_graph()`: builds bipartite graph from stages dict
  - `packages/pivot/src/pivot/cli/targets.py` — The shared `resolve_targets_to_stages()` from Task 1

  **Documentation References**:
  - `packages/pivot/src/pivot/cli/AGENTS.md` — CLI development guidelines (pivot_command decorator, validation patterns)

  **Acceptance Criteria**:

  - [x] `pivot repro output.csv` resolves to the stage that produces `output.csv` and runs it with deps
  - [x] `pivot repro stage_name` still works unchanged
  - [x] `pivot repro` (no args) still runs entire pipeline
  - [x] `pivot repro nonexistent_thing` produces clear error mentioning both stages and output paths
  - [x] The `repro --help` docstring mentions file paths as valid targets
  - [x] `uv run ruff format . && uv run ruff check . && uv run basedpyright` passes

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Existing repro tests still pass
    Tool: Bash (pytest)
    Preconditions: Task 1 complete
    Steps:
      1. uv run pytest packages/pivot/tests/cli/test_repro.py -x -q
      2. Assert: exit code 0, all tests pass
    Expected Result: All existing repro tests pass unchanged
    Evidence: pytest output captured

  Scenario: Code quality passes
    Tool: Bash
    Preconditions: None
    Steps:
      1. uv run ruff format --check packages/pivot/src/pivot/cli/repro.py
      2. uv run ruff check packages/pivot/src/pivot/cli/repro.py
      3. uv run basedpyright packages/pivot/src/pivot/cli/repro.py
      4. Assert: all exit code 0
    Expected Result: Zero errors
    Evidence: Command output captured
  ```

  **Commit**: YES
  - Message: `feat(cli): resolve file paths to stages in pivot repro`
  - Files: `packages/pivot/src/pivot/cli/repro.py`
  - Pre-commit: `uv run pytest packages/pivot/tests/cli/test_repro.py -x -q`

---

- [x] 3. Add tests for file-path resolution in repro

  **What to do**:
  - Add unit tests for `resolve_targets_to_stages()` in `packages/pivot/tests/cli/test_cli_targets.py`
  - Add integration tests for file-path resolution in `packages/pivot/tests/cli/test_repro.py`
  - Test cases:
    1. **Unit: stage name resolves directly** — registered stage name returns in resolved set
    2. **Unit: file path resolves to producer stage** — output file path resolves to the stage that produces it
    3. **Unit: unknown target goes to unresolved** — non-stage, non-output path goes to unresolved list
    4. **Unit: mixed targets** — combination of stage names and file paths both resolve correctly
    5. **Unit: duplicate resolution deduplicates** — two file paths from same stage → one stage in result
    6. **Integration: repro with output file path runs correct stage** — `runner.invoke(cli.cli, ["repro", "output.csv"])` runs the producing stage
    7. **Integration: repro with relative path resolves** — test with `../` style paths
    8. **Integration: repro with unknown file path shows helpful error** — error message mentions output paths

  **Must NOT do**:
  - Do NOT add tests for directory path resolution (out of scope)
  - Do NOT mock internal functions — use real registry with `register_test_stage`
  - Do NOT use `class Test*` structure (project convention: flat functions)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final task)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `packages/pivot/tests/cli/test_cli_targets.py:73-124` — Existing target classification tests showing fixture usage (`mock_discovery`, `set_project_root`), `register_test_stage` pattern, and assertion style
  - `packages/pivot/tests/cli/test_dag_cmd.py:362-382` — `test_dag_target_artifact_path`: integration test for artifact path resolution (the pattern to follow for repro)
  - `packages/pivot/tests/cli/test_repro.py:366-378` — `test_repro_unknown_stage_errors`: existing error path test to complement with file-path error test
  - `packages/pivot/tests/cli/test_repro.py:98-115` — `test_repro_runs_entire_pipeline`: pattern for repro integration tests (runner.invoke, monkeypatch, assertions)

  **Test Infrastructure References**:
  - `packages/pivot/tests/AGENTS.md` — Test conventions: flat functions, module-level helpers, `_helper_` prefix, no lazy imports
  - `packages/pivot/tests/cli/test_dag_cmd.py:22-43` — TypedDict output definitions for test stages
  - `packages/pivot/tests/helpers.py` — `register_test_stage()` function

  **Acceptance Criteria**:

  - [x] All new tests pass: `uv run pytest packages/pivot/tests/cli/test_cli_targets.py packages/pivot/tests/cli/test_repro.py -x -q`
  - [x] Full test suite passes: `uv run pytest packages/pivot/tests -n auto`
  - [x] No decrease in coverage for `cli/targets.py` or `cli/repro.py`

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: New tests pass
    Tool: Bash (pytest)
    Preconditions: Tasks 1 and 2 complete
    Steps:
      1. uv run pytest packages/pivot/tests/cli/test_cli_targets.py packages/pivot/tests/cli/test_repro.py -x -v
      2. Assert: exit code 0
      3. Assert: output contains new test names (resolve_targets_to_stages, repro_file_path)
    Expected Result: All tests pass including new ones
    Evidence: pytest verbose output captured

  Scenario: Full test suite passes
    Tool: Bash (pytest)
    Preconditions: All tasks complete
    Steps:
      1. uv run pytest packages/pivot/tests -n auto -q
      2. Assert: exit code 0, no failures
    Expected Result: Zero test failures
    Evidence: pytest summary output captured

  Scenario: Full quality checks pass
    Tool: Bash
    Preconditions: All tasks complete
    Steps:
      1. uv run ruff format . && uv run ruff check . && uv run basedpyright
      2. Assert: exit code 0
    Expected Result: Zero quality issues
    Evidence: Command output captured
  ```

  **Commit**: YES
  - Message: `test(cli): add tests for file-path resolution in repro`
  - Files: `packages/pivot/tests/cli/test_cli_targets.py`, `packages/pivot/tests/cli/test_repro.py`
  - Pre-commit: `uv run pytest packages/pivot/tests/cli/test_cli_targets.py packages/pivot/tests/cli/test_repro.py -x -q`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `refactor(cli): extract resolve_targets_to_stages to shared targets module` | `cli/targets.py`, `cli/dag.py` | `uv run pytest packages/pivot/tests/cli/test_dag_cmd.py -x -q` |
| 2 | `feat(cli): resolve file paths to stages in pivot repro` | `cli/repro.py` | `uv run pytest packages/pivot/tests/cli/test_repro.py -x -q` |
| 3 | `test(cli): add tests for file-path resolution in repro` | `tests/cli/test_cli_targets.py`, `tests/cli/test_repro.py` | `uv run pytest packages/pivot/tests -n auto` |

---

## Success Criteria

### Verification Commands
```bash
# User's original failing command now works:
# (in a project with stages that produce .jsonl files)
pivot repro --force ../../data/base/raw/agent_runs/*.jsonl
# Expected: resolves to producer stages, runs them with --force

# Stage names still work:
pivot repro my_stage_name
# Expected: unchanged behavior

# Clear error for unknown targets:
pivot repro nonexistent.csv
# Expected: error mentioning both stages and output paths

# Full quality suite:
uv run pytest packages/pivot/tests -n auto
uv run ruff format . && uv run ruff check . && uv run basedpyright
# Expected: all pass
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] All tests pass
- [x] `pivot dag` behavior unchanged (regression check)
