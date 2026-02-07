# Extract TUI into Separate `pivot-tui` Package

## TL;DR

> **Quick Summary**: Extract all Textual-based TUI code from `src/pivot/tui/` into a standalone `pivot-tui` workspace package (`pivot_tui` import). Relocate `console.py` (non-TUI) to its proper home in core. Move `parse_stage_name` to core (used by non-TUI codepaths). Establish a hard dependency boundary: core must never import Textual.
> 
> **Deliverables**:
> - `packages/pivot-tui/` workspace package with all Textual TUI code (including `diff.py` — it's Textual-based)
> - `console.py` relocated to `pivot.cli.console`
> - `parse_stage_name()` moved from `tui/types.py` to `pivot.types` (used unconditionally by non-TUI code)
> - `TuiSink` moved from `engine/sinks.py` to `pivot_tui`
> - CLI lazy-imports from `pivot_tui` with graceful error when not installed
> - `textual` removed from core dependencies
> - TUI tests moved to `packages/pivot-tui/tests/`
> - Import boundary enforced: zero `import textual` in `src/pivot/`
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Task 1/2/3 (parallel) → Task 4 → Task 5/6 → Task 7/8/9

---

## Context

### Original Request
Extract the TUI into an entirely separate Python package to enforce that it's "just one among many clients" and prevent unnecessary coupling between the TUI and core execution logic. Slim down core dependencies.

### Interview Summary
**Key Discussions**:
- PR #395 already created clean seams (StageDataProvider protocol, GraphView adapter, single extraction point) — the extraction builds on this foundation
- `console.py` (Rich/tqdm console output) serves non-TUI CLI commands and must stay in core → `pivot.cli.console`
- `diff.py` is Textual-based (imports `textual.app`, `textual.widgets`, `textual.containers`) — it moves to pivot-tui, NOT core
- `parse_stage_name()` in `tui/types.py` is a pure string utility used unconditionally by `_run_common.sort_for_display()` — must move to core before extraction
- TUI reads lock files, cache, config directly from disk via pivot core modules — keep this pragmatic approach
- In-process TuiSink pattern stays (moves to pivot-tui, owned by the TUI package). RPC-based event streaming is future work
- Pre-alpha: breaking changes OK, no backward compat shims needed

**Research Findings**:
- TUI is ~3,400 LOC across 20 files (run.py 1,316 LOC, diff_panels.py 1,148 LOC, diff.py 342 LOC, widgets/ 1,447 LOC, screens/ 267 LOC, styles/ 271 LOC)
- **diff.py IS Textual**: imports `textual.app`, `textual.binding`, `textual.containers`, `textual.widgets` at lines 6-9. `DiffSummaryPanel` extends `textual.widgets.Static`. Used by `pivot data diff` which must lazy-import from pivot-tui.
- Engine↔TUI uses EventSink/EventSource protocols — already well-abstracted
- TUI-only dependency: `textual>=7.0` (currently polluting core deps)
- uv workspaces is the proven pattern for this (Apache Airflow, AutoGen, Rerun)
- TUI message types in `pivot.types` (TuiLogMessage, TuiStatusMessage, etc.) are plain TypedDicts with no Textual dependency — they stay in core
- TuiUpdate/TuiShutdown are Textual Message classes in `tui/run.py` — they move to pivot-tui along with TuiSink
- `parse_stage_name()` in `tui/types.py` is used by `_run_common.sort_for_display()` at lines 91, 97 — called UNCONDITIONALLY at `repro.py:370` and `repro.py:604` BEFORE the `if tui` branch. Must move to core.

### Metis Review
**Identified Gaps** (addressed):
- CLI UX when pivot-tui not installed → `--tui` flag stays, gives clear "install pivot-tui" error
- Rich stays as core dependency (console.py needs it)
- Need a forbidden-import check enforcing no Textual in core
- Version alignment → pivot-tui pins `pivot` via workspace source in dev
- `from __future__ import annotations` + `TYPE_CHECKING` guards for any remaining Textual type refs
- Import-time side effects → strict lazy import under `--tui` codepaths only

### Momus Review (Round 1)
**Blocking issues found and fixed**:
1. **diff.py is Textual-based** (imports `textual.app`, `textual.widgets`, etc.) — cannot stay in core. Moves to pivot-tui; `pivot data diff` CLI uses lazy import.
2. **`parse_stage_name` used unconditionally** in non-TUI codepath (`_run_common.sort_for_display()` called before `if tui` branch at `repro.py:370` and `repro.py:604`). Must move to core.
3. **Textual version mismatch** — plan said `>=1.0.0`, actual is `>=7.0`. Fixed in Task 3.

---

## Work Objectives

### Core Objective
Establish `pivot-tui` as a standalone uv workspace package that contains all Textual-based TUI code, with a hard one-way dependency: `pivot-tui → pivot` (never reverse).

### Concrete Deliverables
- `packages/pivot-tui/` directory with its own `pyproject.toml`
- `packages/pivot-tui/src/pivot_tui/` containing all moved TUI code (including `diff.py` — it's Textual-based)
- `packages/pivot-tui/tests/` containing TUI-specific tests
- Updated root `pyproject.toml` with workspace config and `[tui]` optional extra
- `console.py` relocated to `pivot.cli.console` (non-TUI, stays in core)
- `parse_stage_name()` moved to `pivot.types` (used unconditionally by non-TUI code)
- All CLI commands working with and without `pivot-tui` installed
- Import boundary test enforcing zero Textual imports in core

### Definition of Done
- [ ] `uv sync --all-packages --active` succeeds
- [ ] `uv run ruff format . && uv run ruff check . && uv run basedpyright` — all clean
- [ ] `uv run pytest tests/ -n auto` — all tests pass
- [ ] `uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto` — all TUI tests pass
- [ ] Zero `import textual` or `from textual` in `src/pivot/` (enforced by script)
- [ ] `pivot --help`, `pivot repro --help`, `pivot run --help` succeed without pivot-tui installed

### Must Have
- One-way dependency: `pivot-tui → pivot`, never reverse
- All Textual imports confined to `packages/pivot-tui/`
- `textual` removed from core `pyproject.toml` dependencies
- CLI works without pivot-tui (non-TUI codepaths unaffected, `--tui` gives helpful error)
- console.py behavior identical after relocation (no formatting changes)
- All existing tests pass

### Must NOT Have (Guardrails)
- No new abstraction layers (client registry, service locator, event bus refactoring)
- No expanding StageDataProvider or adding new protocol surface area
- No RPC/socket decoupling (explicit future work)
- No opportunistic refactors, renames, or formatting-only churn during moves
- No backward compat shims or deprecation warnings (pre-alpha)
- No changes to TUI message types in `pivot.types` (they must not gain Textual imports)
- No new CLI commands or flags beyond making `--tui` give a helpful error when pivot-tui is absent

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks are verifiable WITHOUT any human action. No manual TUI interaction.

### Test Decision
- **Infrastructure exists**: YES (pytest + bun test equivalent via `uv run pytest`)
- **Automated tests**: Tests-after (move existing tests, add import boundary checks)
- **Framework**: pytest (existing)

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Module relocation** | Bash | `python -c "from pivot.cli.console import Console; print('ok')"` |
| **Package structure** | Bash | `uv sync --all-packages --active && uv run --package pivot-tui python -c "import pivot_tui"` |
| **Import boundary** | Bash | Script scanning `src/pivot/` for `import textual` / `from textual` |
| **CLI behavior** | Bash | `uv run pivot --help`, `uv run pivot repro --help` |
| **Test suite** | Bash | `uv run pytest tests/ -n auto`, `uv run --package pivot-tui pytest packages/pivot-tui/tests/` |
| **Type checking** | Bash | `uv run basedpyright` |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Relocate console.py to pivot.cli.console
├── Task 2: Move parse_stage_name to pivot.types
└── Task 3: Set up uv workspace + pivot-tui skeleton

Wave 2 (After Wave 1):
└── Task 4: Move all TUI code to pivot-tui (includes diff.py — it's Textual-based)

Wave 3 (After Wave 2):
├── Task 5: Extract TuiSink to pivot-tui
└── Task 6: Update CLI integration (lazy imports, including data.py for diff)

Wave 4 (After Wave 3):
├── Task 7: Clean up old pivot.tui + update dependencies
├── Task 8: Move TUI tests to pivot-tui
└── Task 9: Tooling, import boundary enforcement, full verification

Critical Path: Task 3 → Task 4 → Task 5/6 → Task 7/9
Parallel Speedup: ~35% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 4 | 2, 3 |
| 2 | None | 4 | 1, 3 |
| 3 | None | 4 | 1, 2 |
| 4 | 1, 2, 3 | 5, 6, 7, 8 | None |
| 5 | 4 | 7, 9 | 6 |
| 6 | 4 | 7, 9 | 5 |
| 7 | 5, 6 | 9 | 8 |
| 8 | 4 | 9 | 7 |
| 9 | 7, 8 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2, 3 | Three parallel `quick` agents |
| 2 | 4 | One `unspecified-high` agent (biggest task) |
| 3 | 5, 6 | Two parallel `quick` agents |
| 4 | 7, 8, 9 | Three parallel agents (9 is `unspecified-high`) |

---

## TODOs

- [x] 1. Relocate `console.py` to `pivot.cli.console`

  **What to do**:
  - Use `lsp_find_references` on `pivot.tui.console` and the `Console` class to enumerate all import sites before moving
  - Move `src/pivot/tui/console.py` → `src/pivot/cli/console.py`
  - Update all imports project-wide:
    - `src/pivot/cli/repro.py`: `from pivot.tui import console` → `from pivot.cli import console`
    - `src/pivot/cli/run.py`: `from pivot.tui import console` → `from pivot.cli import console`
    - `src/pivot/cli/status.py`: `from pivot.tui import console` → `from pivot.cli import console`
    - Any other files found by `lsp_find_references`
  - Update `src/pivot/tui/__init__.py`: remove `console` import/export
  - Within `console.py` itself: update any `from pivot.tui.X` imports to their new locations (if any internal cross-references exist — there should be very few since console.py mostly imports from `pivot.types`)
  - Run tests to verify

  **Must NOT do**:
  - Do not change any console output formatting or behavior
  - Do not refactor console.py internals
  - Do not change the Console class API

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file move with mechanical import updates
  - **Skills**: []
    - No special skills needed; standard file operations + LSP

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/pivot/tui/console.py` — The file being moved. Contains `Console` class with Rich/tqdm output methods
  - `src/pivot/tui/__init__.py` — Currently exports console; must be updated to remove it

  **API/Type References**:
  - `src/pivot/types.py` — Console imports `StageStatus`, `DisplayCategory`, `MetricValue` from here. These imports stay unchanged.

  **Import Sites** (known from research, verify with `lsp_find_references`):
  - `src/pivot/cli/repro.py` — `from pivot.tui import console`
  - `src/pivot/cli/run.py` — `from pivot.tui import console`
  - `src/pivot/cli/status.py` — `from pivot.tui import console`

  **Test References**:
  - Run `uv run pytest tests/ -n auto -x` to verify nothing breaks

  **Acceptance Criteria**:

  - [ ] `src/pivot/cli/console.py` exists with identical content to original
  - [ ] `src/pivot/tui/console.py` no longer exists
  - [ ] No occurrences of `pivot.tui.console` in `src/pivot/` (verify with `grep -r "pivot.tui.console" src/pivot/` → empty)
  - [ ] `uv run python -c "from pivot.cli.console import Console; print('ok')"` → prints `ok`
  - [ ] `uv run pytest tests/ -n auto -x` → all tests pass
  - [ ] `uv run ruff check src/` → clean

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Console import works at new location
    Tool: Bash
    Preconditions: uv sync completed
    Steps:
      1. uv run python -c "from pivot.cli.console import Console; print('ok')"
      2. Assert: stdout contains "ok", exit code 0
    Expected Result: Console class importable from new location
    Evidence: Command output captured

  Scenario: Old import location is gone
    Tool: Bash (grep)
    Preconditions: File move completed
    Steps:
      1. grep -r "pivot\.tui\.console" src/pivot/ --include="*.py"
      2. Assert: no output (empty), exit code 1
      3. grep -r "from pivot.tui import console" src/pivot/ --include="*.py"
      4. Assert: no output (empty), exit code 1
    Expected Result: No references to old location remain
    Evidence: grep output captured

  Scenario: CLI commands still work
    Tool: Bash
    Steps:
      1. uv run pivot status --help
      2. Assert: exit code 0, help text displayed
    Expected Result: Status command (which uses Console) works
    Evidence: Command output captured
  ```

  **Commit**: YES
  - Message: `refactor(tui): move console.py to pivot.cli.console`
  - Files: `src/pivot/cli/console.py`, `src/pivot/tui/console.py` (deleted), import updates
  - Pre-commit: `uv run pytest tests/ -n auto -x`

---

- [x] 2. Move `parse_stage_name` from `tui/types.py` to `pivot.types`

  **What to do**:
  - `parse_stage_name()` is a pure string utility (splits on "@", 5 lines, zero TUI dependency) currently in `src/pivot/tui/types.py:13-18`. It's used unconditionally by `_run_common.sort_for_display()` (lines 84, 91, 97) which is called BEFORE the `if tui` branch at `repro.py:370` and `repro.py:604`. If it stays in `tui/types.py` and moves to pivot-tui, non-TUI runs will crash when pivot-tui isn't installed.
  - Move the `parse_stage_name` function from `src/pivot/tui/types.py` to `src/pivot/types.py`
  - Update all import sites:
    - `src/pivot/cli/_run_common.py:84`: `from pivot.tui.types import parse_stage_name` → `from pivot.types import parse_stage_name`
    - `src/pivot/tui/types.py` itself: the `StageInfo.__post_init__` at line 97 calls `parse_stage_name(self.name)`. After the function moves, this file will need `from pivot.types import parse_stage_name` (or it will get it when the whole types.py moves to pivot-tui in Task 4, since pivot-tui imports from pivot.types)
    - Use `lsp_find_references` to find any other callers
  - Remove `parse_stage_name` from `src/pivot/tui/types.py` and add an import from `pivot.types` so existing callers within `tui/types.py` still work until Task 4 moves everything
  - Run tests to verify

  **Must NOT do**:
  - Do not change parse_stage_name behavior (it's a trivial string split)
  - Do not move other types or functions from tui/types.py (those move in Task 4)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single function relocation with ~2 import site updates
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/pivot/tui/types.py:13-18` — The function being moved. Pure string utility: `if "@" in name: base, variant = name.split("@", 1); return (base, variant); return (name, "")`
  - `src/pivot/types.py` — Destination. Already contains StageStatus, DisplayCategory, and other shared types.

  **Import Sites** (verify with `lsp_find_references`):
  - `src/pivot/cli/_run_common.py:84` — `from pivot.tui.types import parse_stage_name`
  - `src/pivot/tui/types.py:97` — `StageInfo.__post_init__` calls `parse_stage_name(self.name)` (internal usage within same file)

  **WHY Each Reference Matters**:
  - `_run_common.py` — This is the CRITICAL caller: `sort_for_display()` runs unconditionally in non-TUI codepaths. Import must point to core.
  - `tui/types.py` — Internal usage; after Task 4 moves this file to pivot-tui, it'll import from `pivot.types` naturally.

  **Acceptance Criteria**:

  - [ ] `parse_stage_name` function exists in `src/pivot/types.py`
  - [ ] `parse_stage_name` function removed from `src/pivot/tui/types.py` (replaced with import from `pivot.types`)
  - [ ] `src/pivot/cli/_run_common.py` imports `parse_stage_name` from `pivot.types`
  - [ ] `uv run python -c "from pivot.types import parse_stage_name; print(parse_stage_name('train@gpu'))"` → prints `('train', 'gpu')`
  - [ ] `uv run pytest tests/ -n auto -x` → all tests pass
  - [ ] `uv run ruff check src/` → clean

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: parse_stage_name importable from pivot.types
    Tool: Bash
    Steps:
      1. uv run python -c "from pivot.types import parse_stage_name; print(parse_stage_name('train@gpu'))"
      2. Assert: stdout "('train', 'gpu')", exit code 0
      3. uv run python -c "from pivot.types import parse_stage_name; print(parse_stage_name('simple'))"
      4. Assert: stdout "('simple', '')", exit code 0
    Expected Result: Function works at new location
    Evidence: Command output

  Scenario: _run_common no longer imports from tui
    Tool: Bash (grep)
    Steps:
      1. grep -n "pivot.tui" src/pivot/cli/_run_common.py
      2. Assert: no matches for parse_stage_name import (TYPE_CHECKING imports of MessagePoster may still exist — those are handled in Task 6)
    Expected Result: parse_stage_name import points to pivot.types
    Evidence: grep output

  Scenario: Existing tui/types.py still works (temporary compat)
    Tool: Bash
    Steps:
      1. uv run python -c "from pivot.tui.types import parse_stage_name; print('ok')"
      2. Assert: stdout "ok" (re-export from pivot.types)
    Expected Result: Backward compat until Task 4 moves the file
    Evidence: Command output
  ```

  **Commit**: YES
  - Message: `refactor: move parse_stage_name to pivot.types (needed by non-TUI codepaths)`
  - Files: `src/pivot/types.py`, `src/pivot/tui/types.py`, `src/pivot/cli/_run_common.py`
  - Pre-commit: `uv run pytest tests/ -n auto -x`

---

- [x] 3. Set up uv workspace + pivot-tui package skeleton

  **What to do**:
  - Create directory structure:
    ```
    packages/
    └── pivot-tui/
        ├── src/
        │   └── pivot_tui/
        │       └── __init__.py  (empty or minimal, placeholder)
        ├── tests/
        │   └── __init__.py
        └── pyproject.toml
    ```
  - Create `packages/pivot-tui/pyproject.toml`:
    ```toml
    [project]
    name = "pivot-tui"
    version = "0.1.0"
    description = "Textual-based TUI for the Pivot pipeline framework"
    requires-python = ">=3.13,<3.14"
    dependencies = [
        "pivot",
        "textual>=7.0",
    ]

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

    [tool.hatch.build.targets.wheel]
    packages = ["src/pivot_tui"]
    ```
  - Update root `pyproject.toml`:
    - Add `[tool.uv.workspace]` section with `members = ["packages/*"]`
    - Add `[tool.uv.sources]` section: `pivot-tui = { workspace = true }`
    - Add optional extra: `[project.optional-dependencies]` with `tui = ["pivot-tui"]`
    - Do NOT remove textual from core deps yet (that happens in Task 7 after code is moved)
  - Run `uv sync --all-packages --active` to verify workspace resolves
  - Verify `uv run --package pivot-tui python -c "import pivot_tui; print('ok')"` works

  **Must NOT do**:
  - Do not move any TUI code yet (that's Task 4)
  - Do not modify existing package structure beyond workspace config
  - Do not change core dependencies yet

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Scaffolding with known structure, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `pyproject.toml` (root) — Current package config, build system, dependencies. Read this fully to understand current structure before modifying.
  - uv workspace docs: `[tool.uv.workspace]` with `members` glob, `[tool.uv.sources]` with `workspace = true`

  **Documentation References**:
  - uv workspace documentation: https://docs.astral.sh/uv/concepts/workspaces/

  **WHY Each Reference Matters**:
  - Root `pyproject.toml` — Must understand existing build system (hatchling), dependency list, and optional-dependencies groups before adding workspace config
  - uv docs — Correct syntax for workspace members and source declarations

  **Acceptance Criteria**:

  - [ ] `packages/pivot-tui/pyproject.toml` exists with correct metadata
  - [ ] `packages/pivot-tui/src/pivot_tui/__init__.py` exists
  - [ ] Root `pyproject.toml` has `[tool.uv.workspace]` section
  - [ ] `uv sync --all-packages --active` succeeds (exit code 0)
  - [ ] `uv run --package pivot-tui python -c "import pivot_tui; print('ok')"` → prints `ok`

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Workspace resolves correctly
    Tool: Bash
    Steps:
      1. uv sync --all-packages --active
      2. Assert: exit code 0, no errors
    Expected Result: Both packages install in workspace
    Evidence: Command output

  Scenario: pivot-tui package importable
    Tool: Bash
    Steps:
      1. uv run --package pivot-tui python -c "import pivot_tui; print('ok')"
      2. Assert: stdout "ok", exit code 0
    Expected Result: Empty package skeleton importable
    Evidence: Command output

  Scenario: Core package still works
    Tool: Bash
    Steps:
      1. uv run pivot --help
      2. Assert: exit code 0
      3. uv run pytest tests/ -n auto -x --timeout=60
      4. Assert: all tests pass
    Expected Result: Adding workspace doesn't break existing functionality
    Evidence: Command output, test results
  ```

  **Commit**: YES
  - Message: `build: set up uv workspace with pivot-tui package skeleton`
  - Files: `packages/pivot-tui/`, `pyproject.toml` (root)
  - Pre-commit: `uv sync --all-packages --active && uv run pytest tests/ -n auto -x`

---

- [x] 4. Move all TUI code to `pivot-tui` package

  **What to do**:
  - This is the main extraction task. Move all Textual-based TUI files from `src/pivot/tui/` to `packages/pivot-tui/src/pivot_tui/`.
  - **Files to move** (preserving directory structure):
    - `src/pivot/tui/run.py` → `packages/pivot-tui/src/pivot_tui/run.py`
    - `src/pivot/tui/types.py` → `packages/pivot-tui/src/pivot_tui/types.py`
    - `src/pivot/tui/diff.py` → `packages/pivot-tui/src/pivot_tui/diff.py` (**Note**: diff.py IS Textual-based — imports `textual.app`, `textual.widgets`, `textual.containers`, `textual.binding` at lines 6-9)
    - `src/pivot/tui/diff_panels.py` → `packages/pivot-tui/src/pivot_tui/diff_panels.py`
    - `src/pivot/tui/stats.py` → `packages/pivot-tui/src/pivot_tui/stats.py`
    - `src/pivot/tui/rpc_client.py` → `packages/pivot-tui/src/pivot_tui/rpc_client.py`
    - `src/pivot/tui/widgets/` → `packages/pivot-tui/src/pivot_tui/widgets/` (all files: `__init__.py`, `stage_list.py`, `logs.py`, `panels.py`, `status.py`, `footer.py`, `debug.py`)
    - `src/pivot/tui/screens/` → `packages/pivot-tui/src/pivot_tui/screens/` (all files: `__init__.py`, `history_list.py`, `confirm_dialog.py`, `help.py`)
    - `src/pivot/tui/styles/` → `packages/pivot-tui/src/pivot_tui/styles/` (all files: `__init__.py`, `pivot.tcss`, `modal.tcss`)
  - **Update all intra-TUI imports** within moved files. Use `ast_grep_search` to find all `from pivot.tui` patterns. Change:
    - `from pivot.tui.widgets import X` → `from pivot_tui.widgets import X` (or relative: `from .widgets import X`)
    - `from pivot.tui.screens import X` → `from pivot_tui.screens import X`
    - `from pivot.tui.types import X` → `from pivot_tui.types import X`
    - `from pivot.tui.diff_panels import X` → `from pivot_tui.diff_panels import X`
    - `from pivot.tui.stats import X` → `from pivot_tui.stats import X`
    - `from pivot.tui import X` → `from pivot_tui import X`
    - etc. for all `pivot.tui.*` references within the moved files
  - **Imports from core stay unchanged**: `from pivot.types import StageStatus` etc. remain as-is (pivot-tui depends on pivot core)
  - **Create `pivot_tui/__init__.py`** with appropriate exports. At minimum:
    ```python
    from pivot_tui import diff as diff
    from pivot_tui import diff_panels as diff_panels
    ```
    The `run` module should NOT be imported at package level (to avoid pulling in Textual at import time when only types are needed). Keep the same pattern as the original `tui/__init__.py` which avoided importing `run`.
  - **Do NOT update CLI imports yet** (that's Task 6). At this point, CLI imports will be broken — that's expected and will be fixed in Tasks 5-6.
  - Use `ast_grep_search` with pattern `from pivot.tui` to verify no stale references remain within the moved files

  **Must NOT do**:
  - Do not refactor any TUI code while moving it
  - Do not change any widget/screen behavior
  - Do not add new exports or change the public API
  - Do not update CLI/engine imports (Tasks 5-6)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Large mechanical task (~15 files, many import updates), requires careful attention to avoid missing references
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (solo)
  - **Blocks**: Tasks 5, 6, 7, 8
  - **Blocked By**: Tasks 1, 2, 3

  **References**:

  **Pattern References**:
  - `src/pivot/tui/__init__.py` — Current exports: console (moved in T1), diff (moved in T2), diff_panels. Note: run is NOT imported here to avoid circular imports. Follow same pattern.
  - `src/pivot/tui/run.py` — Main TUI app (1,316 LOC). Defines `PivotApp`, `MessagePoster` protocol, `TuiUpdate`/`TuiShutdown` message classes, `format_reload_summary()`. Imports from: `pivot.types`, `pivot.executor`, `pivot.storage`, `pivot.config`, `pivot.parameters`, `pivot.project`, `pivot.explain`, `pivot.show`, and multiple `pivot.tui.*` submodules.
  - `src/pivot/tui/types.py` — Defines `StageDataProvider` protocol, `StageInfo`, `LogEntry`, `ExecutionHistoryEntry`, `PendingHistoryState` TypedDicts. Imports: `pivot.types.StageStatus`, `pivot.registry.RegistryStageInfo` (TYPE_CHECKING only).
  - `src/pivot/tui/diff.py` — Textual-based data diff display (342 LOC). Imports `textual.app`, `textual.widgets`, `textual.containers`, `textual.binding` (lines 6-9). `DiffSummaryPanel` extends `textual.widgets.Static`. Also imports from `pivot.project`, `pivot.show.data`, `pivot.types`.
  - `src/pivot/tui/diff_panels.py` — Input/Output diff panels (1,148 LOC). Heavy imports from `pivot.storage.lock`, `pivot.storage.cache`, `pivot.show.data`, `pivot.show.metrics`, `pivot.explain`, `pivot.types`.
  - `src/pivot/tui/rpc_client.py` — Lightweight RPC client (~67 LOC). Imports only stdlib (asyncio, json, pathlib). Self-contained.
  - `src/pivot/tui/stats.py` — Debug stats (~121 LOC). Minimal imports.
  - `src/pivot/tui/widgets/__init__.py` — Widget exports. Check what's exported and maintain it.
  - `src/pivot/tui/screens/__init__.py` — Screen exports.

  **Acceptance Criteria**:

  - [ ] All listed files exist under `packages/pivot-tui/src/pivot_tui/`
  - [ ] No Textual-related `.py` files remain in `src/pivot/tui/` (console.py was moved in T1, diff.py in T2, everything else moves here)
  - [ ] Zero `from pivot.tui` imports within `packages/pivot-tui/` (all updated to `pivot_tui`)
  - [ ] `uv run --package pivot-tui python -c "import pivot_tui; print('ok')"` → prints `ok`
  - [ ] `uv run --package pivot-tui python -c "from pivot_tui.types import StageDataProvider; print('ok')"` → prints `ok`
  - [ ] `uv run --package pivot-tui python -c "from pivot_tui.rpc_client import send_run_command; print('ok')"` → prints `ok`
  - [ ] `uv run ruff check packages/pivot-tui/src/` → clean

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All TUI modules importable from new location
    Tool: Bash
    Steps:
      1. uv run --package pivot-tui python -c "
         from pivot_tui import diff_panels
         from pivot_tui.types import StageDataProvider, StageInfo, LogEntry
         from pivot_tui.rpc_client import send_run_command
         from pivot_tui.stats import DebugStats
         from pivot_tui.widgets import StageListView
         from pivot_tui.screens import HistoryScreen
         print('all imports ok')
         "
      2. Assert: stdout "all imports ok", exit code 0
    Expected Result: All moved modules importable
    Evidence: Command output

  Scenario: No stale pivot.tui references in moved code
    Tool: Bash (grep/ast_grep_search)
    Steps:
      1. grep -rn "from pivot\.tui" packages/pivot-tui/src/ --include="*.py"
      2. Assert: empty output, exit code 1
      3. grep -rn "import pivot\.tui" packages/pivot-tui/src/ --include="*.py"
      4. Assert: empty output, exit code 1
    Expected Result: Zero references to old pivot.tui within pivot-tui code
    Evidence: grep output

  Scenario: Core pivot imports still work within TUI code
    Tool: Bash
    Steps:
      1. uv run --package pivot-tui python -c "
         from pivot_tui.types import StageDataProvider
         from pivot.types import StageStatus
         print('cross-package imports ok')
         "
      2. Assert: stdout "cross-package imports ok", exit code 0
    Expected Result: pivot-tui can import from pivot core
    Evidence: Command output

  Scenario: Styles and CSS files moved correctly
    Tool: Bash
    Steps:
      1. ls packages/pivot-tui/src/pivot_tui/styles/pivot.tcss
      2. Assert: file exists, exit code 0
      3. ls packages/pivot-tui/src/pivot_tui/styles/modal.tcss
      4. Assert: file exists, exit code 0
    Expected Result: Textual CSS files present
    Evidence: ls output
  ```

  **Commit**: YES
  - Message: `refactor(tui): move all Textual TUI code to pivot-tui package`
  - Files: All moved files in `packages/pivot-tui/`, deleted originals in `src/pivot/tui/`
  - Pre-commit: `uv run ruff check packages/pivot-tui/src/`

---

- [x] 5. Extract TuiSink from `engine/sinks.py` to `pivot-tui`

  **What to do**:
  - Use `lsp_find_references` on `TuiSink` class to find all usage sites
  - Read `src/pivot/engine/sinks.py` to understand TuiSink class boundaries (lines ~145-191 per research)
  - Extract `TuiSink` class from `src/pivot/engine/sinks.py` into `packages/pivot-tui/src/pivot_tui/sink.py`
  - TuiSink currently imports:
    - `MessagePoster` from `pivot.tui.run` → now `pivot_tui.run`
    - `TuiUpdate`, `TuiShutdown` from `pivot.tui.run` → now `pivot_tui.run`
    - `OutputEvent` types from `pivot.engine.types`
    - TUI message types from `pivot.types` (TuiLogMessage, TuiStatusMessage, etc.)
  - Update imports within the new `sink.py`:
    - `from pivot_tui.run import MessagePoster, TuiUpdate, TuiShutdown`
    - `from pivot.engine.types import OutputEvent` (cross-package, stays)
    - `from pivot.types import TuiLogMessage, ...` (stays)
  - Remove TuiSink class from `src/pivot/engine/sinks.py` (keep ConsoleSink, ResultCollectorSink)
  - Remove any TUI-related imports from `engine/sinks.py` that are no longer needed
  - Update `src/pivot/cli/_run_common.py` where TuiSink is instantiated: change import from `pivot.engine.sinks` to `pivot_tui.sink` (lazy import under `tui=True` branch)
  - Export TuiSink from `pivot_tui/__init__.py` or `pivot_tui/sink.py`

  **Must NOT do**:
  - Do not change TuiSink behavior or API
  - Do not modify ConsoleSink or ResultCollectorSink
  - Do not change the EventSink protocol

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single class extraction with focused import updates
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 6)
  - **Blocks**: Tasks 7, 9
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `src/pivot/engine/sinks.py` — Contains TuiSink (lines ~145-191), ConsoleSink, ResultCollectorSink. Read the full file to understand class boundaries and shared imports.
  - `src/pivot/engine/types.py` — Defines `EventSink` protocol and `OutputEvent` type union. TuiSink implements EventSink.

  **Import Sites** (verify with `lsp_find_references`):
  - `src/pivot/cli/_run_common.py` — Creates TuiSink instance, passes to engine

  **WHY Each Reference Matters**:
  - `sinks.py` — Need to understand which imports are shared with ConsoleSink vs TuiSink-specific, to remove only TuiSink-specific imports
  - `_run_common.py` — The only place that creates TuiSink; import must change to `pivot_tui.sink`

  **Acceptance Criteria**:

  - [ ] `packages/pivot-tui/src/pivot_tui/sink.py` exists with TuiSink class
  - [ ] `src/pivot/engine/sinks.py` no longer contains TuiSink class
  - [ ] `src/pivot/engine/sinks.py` has no `from pivot.tui` or `from pivot_tui` imports
  - [ ] `uv run --package pivot-tui python -c "from pivot_tui.sink import TuiSink; print('ok')"` → prints `ok`
  - [ ] `uv run ruff check src/pivot/engine/sinks.py` → clean
  - [ ] `uv run ruff check packages/pivot-tui/src/pivot_tui/sink.py` → clean

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: TuiSink importable from pivot-tui
    Tool: Bash
    Steps:
      1. uv run --package pivot-tui python -c "from pivot_tui.sink import TuiSink; print('ok')"
      2. Assert: stdout "ok", exit code 0
    Expected Result: TuiSink lives in pivot-tui now
    Evidence: Command output

  Scenario: Engine sinks module clean of TUI references
    Tool: Bash (grep)
    Steps:
      1. grep -n "pivot.tui\|pivot_tui\|TuiSink\|TuiUpdate\|TuiShutdown\|MessagePoster" src/pivot/engine/sinks.py
      2. Assert: empty output
    Expected Result: No TUI references remain in engine sinks
    Evidence: grep output

  Scenario: Core engine sinks still work
    Tool: Bash
    Steps:
      1. uv run python -c "from pivot.engine.sinks import ConsoleSink, ResultCollectorSink; print('ok')"
      2. Assert: stdout "ok", exit code 0
    Expected Result: Other sinks unaffected
    Evidence: Command output
  ```

  **Commit**: YES (groups with Task 6)
  - Message: `refactor(tui): extract TuiSink from engine/sinks.py to pivot-tui`
  - Files: `packages/pivot-tui/src/pivot_tui/sink.py`, `src/pivot/engine/sinks.py`, `src/pivot/cli/_run_common.py`
  - Pre-commit: `uv run ruff check src/ packages/`

---

- [x] 6. Update CLI integration — lazy imports from `pivot_tui`

  **What to do**:
  - Use `ast_grep_search` to find all `from pivot.tui` and `import pivot.tui` patterns in `src/pivot/cli/`
  - Update each CLI module that imports from the old `pivot.tui` location to use lazy imports from `pivot_tui`:
    - `src/pivot/cli/repro.py` — imports `pivot.tui.run` (PivotApp, format_reload_summary). Change to lazy import of `pivot_tui.run` within the `--tui` codepath.
    - `src/pivot/cli/run.py` — imports `pivot.tui.run` (PivotApp). Same pattern.
    - `src/pivot/cli/data.py` — imports `pivot.tui.diff` for the `pivot data diff` command. Change to lazy import of `pivot_tui.diff`. The `data diff` subcommand should give a clear "install pivot-tui" error if pivot-tui is not installed.
    - `src/pivot/cli/_run_common.py` — imports `pivot.tui.run.MessagePoster` (TYPE_CHECKING only — already lazy). `parse_stage_name` import was already moved to `pivot.types` in Task 2. Verify no remaining `pivot.tui` imports exist.
  - **Add graceful error handling**: When `--tui` flag is used but `pivot-tui` is not installed, catch `ImportError` and provide a clear message:
    ```python
    try:
        from pivot_tui.run import PivotApp
    except ImportError:
        raise click.UsageError(
            "The TUI requires the 'pivot-tui' package. "
            "Install it with: uv pip install pivot-tui"
        )
    ```
  - **Ensure no top-level imports**: All `pivot_tui` imports must be inside function bodies or guarded by `TYPE_CHECKING`. The CLI modules must be importable without pivot-tui installed.
  - **Update `engine/sinks.py`** if there are any remaining TUI imports (should have been cleaned in Task 5, but verify)
  - Run `uv run pivot --help` and `uv run pivot repro --help` to verify CLI loads without errors

  **Must NOT do**:
  - Do not add new CLI commands or flags
  - Do not change CLI behavior beyond the import paths
  - Do not add backward compat shims or deprecation warnings

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Focused import updates in ~3 CLI files
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 5)
  - **Blocks**: Tasks 7, 9
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `src/pivot/cli/repro.py` — The `--tui` codepath that instantiates PivotApp
  - `src/pivot/cli/run.py` — The `--tui` codepath for single-stage runs
  - `src/pivot/cli/_run_common.py` — Shared helper that configures sinks (TuiSink) and uses MessagePoster, parse_stage_name

  **Import Sites** (exhaustive list from research):
  - `src/pivot/cli/repro.py`: `from pivot.tui import console` (already moved in T1), `from pivot.tui import run as tui_run`
  - `src/pivot/cli/run.py`: `from pivot.tui import console` (already moved in T1), `from pivot.tui import run as tui_run`
  - `src/pivot/cli/data.py`: `from pivot.tui import diff` or similar — used by `pivot data diff` command. Must change to lazy import from `pivot_tui.diff`.
  - `src/pivot/cli/_run_common.py`: `from pivot.tui.run import MessagePoster` (TYPE_CHECKING only), `from pivot.tui.types import parse_stage_name` (already moved to `pivot.types` in Task 2)
  - `src/pivot/engine/sinks.py`: should be clean after Task 5

  **WHY Each Reference Matters**:
  - These are the REVERSE COUPLING points — core importing from TUI. Every one must be changed to lazy imports from `pivot_tui` to establish the one-way dependency boundary.

  **Acceptance Criteria**:

  - [ ] Zero `from pivot.tui` imports in `src/pivot/` (verified by grep)
  - [ ] Zero top-level `from pivot_tui` imports in `src/pivot/cli/` (all must be inside functions or TYPE_CHECKING)
  - [ ] `uv run pivot --help` → succeeds (exit code 0)
  - [ ] `uv run pivot repro --help` → succeeds (exit code 0)
  - [ ] `uv run pivot run --help` → succeeds (exit code 0)
  - [ ] `uv run ruff check src/pivot/cli/` → clean

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: No pivot.tui imports remain in core
    Tool: Bash (grep)
    Steps:
      1. grep -rn "from pivot\.tui" src/pivot/ --include="*.py"
      2. Assert: empty output, exit code 1
      3. grep -rn "import pivot\.tui" src/pivot/ --include="*.py"
      4. Assert: empty output, exit code 1
    Expected Result: Zero references to old pivot.tui package in core
    Evidence: grep output

  Scenario: No top-level pivot_tui imports in CLI
    Tool: Bash
    Steps:
      1. For each file in src/pivot/cli/*.py, verify no top-level 'from pivot_tui' or 'import pivot_tui' (outside of TYPE_CHECKING blocks or function bodies)
      2. Use ast_grep_search to find top-level pivot_tui imports
    Expected Result: All pivot_tui imports are lazy (inside functions)
    Evidence: ast_grep output

  Scenario: CLI help commands work
    Tool: Bash
    Steps:
      1. uv run pivot --help
      2. Assert: exit code 0, shows help
      3. uv run pivot repro --help
      4. Assert: exit code 0, shows repro options
      5. uv run pivot run --help
      6. Assert: exit code 0, shows run options
    Expected Result: All CLI commands load without importing Textual
    Evidence: Command outputs
  ```

  **Commit**: YES (groups with Task 5)
  - Message: `refactor(cli): update TUI imports to lazy-load from pivot-tui package`
  - Files: `src/pivot/cli/repro.py`, `src/pivot/cli/run.py`, `src/pivot/cli/_run_common.py`
  - Pre-commit: `uv run pivot --help && uv run ruff check src/pivot/cli/`

---

- [ ] 7. Clean up old `pivot.tui` + update dependencies

  **What to do**:
  - **Remove the old `src/pivot/tui/` directory** entirely. After Tasks 1-6:
    - `console.py` was moved to `pivot.cli.console` (Task 1)
    - `parse_stage_name` was moved to `pivot.types` (Task 2)
    - All Textual files (including `diff.py`) moved to `pivot_tui` (Task 4)
    - What might remain: `__init__.py`, possibly empty dirs
    - Remove the entire `src/pivot/tui/` directory
  - **Update root `pyproject.toml` dependencies**:
    - Remove `textual` (and version constraint) from `[project.dependencies]`
    - Keep `rich` in core dependencies (needed by `pivot.cli.console`)
    - Keep `tqdm` in core dependencies (needed by `pivot.cli.console`)
    - Verify `[project.optional-dependencies]` has `tui = ["pivot-tui"]` (added in Task 3)
  - **Verify core works without Textual**:
    - `uv sync --active` (core only, without pivot-tui)
    - `uv run python -c "import pivot; import pivot.cli; import pivot.engine; print('ok')"` → must work
  - **Verify `pivot[tui]` installs both packages**:
    - Confirm that `uv pip install pivot[tui]` would pull in `pivot-tui`

  **Must NOT do**:
  - Do not change any other dependencies
  - Do not remove `rich` or `tqdm` from core (console.py needs them)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Cleanup and config updates, small scope
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 8)
  - **Blocks**: Task 9
  - **Blocked By**: Tasks 5, 6

  **References**:

  **Pattern References**:
  - `pyproject.toml` (root) — Dependencies section, optional-dependencies

  **Acceptance Criteria**:

  - [ ] `src/pivot/tui/` directory does not exist
  - [ ] `textual` not in root `pyproject.toml` `[project.dependencies]`
  - [ ] `rich` still in root `pyproject.toml` `[project.dependencies]`
  - [ ] `tui = ["pivot-tui"]` in `[project.optional-dependencies]`
  - [ ] `uv sync --active` succeeds (core only)
  - [ ] `uv run python -c "import pivot; import pivot.cli; import pivot.engine; print('ok')"` → prints `ok`

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Core installs and imports without Textual
    Tool: Bash
    Steps:
      1. uv sync --active
      2. Assert: exit code 0
      3. uv run python -c "import pivot; import pivot.cli; import pivot.engine; print('ok')"
      4. Assert: stdout "ok", exit code 0
    Expected Result: Core package works without textual installed
    Evidence: Command output

  Scenario: Old TUI directory is gone
    Tool: Bash
    Steps:
      1. ls src/pivot/tui/ 2>/dev/null
      2. Assert: exit code non-zero (directory doesn't exist)
    Expected Result: Clean removal
    Evidence: ls output

  Scenario: Textual not in core deps
    Tool: Bash (grep)
    Steps:
      1. grep -n "textual" pyproject.toml
      2. Assert: only appears in [project.optional-dependencies] tui group or workspace sources, NOT in [project.dependencies]
    Expected Result: textual fully removed from core deps
    Evidence: grep output
  ```

  **Commit**: YES
  - Message: `refactor(tui): remove old pivot.tui directory, drop textual from core deps`
  - Files: `src/pivot/tui/` (deleted), `pyproject.toml`
  - Pre-commit: `uv sync --active && uv run python -c "import pivot; print('ok')"`

---

- [ ] 8. Move TUI tests to `pivot-tui` package

  **What to do**:
  - Identify all TUI-specific test files:
    - `tests/tui/test_tui_integration.py` → `packages/pivot-tui/tests/test_tui_integration.py`
    - `tests/engine/test_tui_sink.py` → `packages/pivot-tui/tests/test_tui_sink.py` (since TuiSink now lives in pivot-tui)
    - `tests/integration/test_tui_force_rerun.py` — This is an integration test that tests engine+TUI together. It may need to stay in root OR move to pivot-tui with appropriate fixtures.
  - Use `grep` to find all test files that import from `pivot.tui` or reference TUI functionality
  - Update imports in moved test files:
    - `from pivot.tui.run import X` → `from pivot_tui.run import X`
    - `from pivot.tui.types import X` → `from pivot_tui.types import X`
    - `from pivot.engine.sinks import TuiSink` → `from pivot_tui.sink import TuiSink`
  - Configure test discovery for pivot-tui:
    - Add `[tool.pytest.ini_options]` to `packages/pivot-tui/pyproject.toml` if needed
    - Or configure root `pyproject.toml` pytest settings to discover both test directories
  - Verify both test suites run independently:
    - `uv run pytest tests/ -n auto` (core tests)
    - `uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto` (TUI tests)

  **Must NOT do**:
  - Do not rewrite test logic, only update imports
  - Do not delete integration tests that belong in the root test suite

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical file moves and import updates in test files
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `tests/tui/test_tui_integration.py` — Main TUI integration test
  - `tests/engine/test_tui_sink.py` — TuiSink unit tests
  - `tests/integration/test_tui_force_rerun.py` — Engine+TUI integration test

  **Test References**:
  - Run both test suites to verify: `uv run pytest tests/ -n auto` and `uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto`

  **Acceptance Criteria**:

  - [ ] TUI test files exist under `packages/pivot-tui/tests/`
  - [ ] No `from pivot.tui` imports in any test files
  - [ ] `uv run pytest tests/ -n auto` → core tests pass (TUI tests no longer here)
  - [ ] `uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto` → TUI tests pass

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Core tests pass without TUI tests
    Tool: Bash
    Steps:
      1. uv run pytest tests/ -n auto -x
      2. Assert: exit code 0, all tests pass
    Expected Result: Core test suite is self-contained
    Evidence: pytest output

  Scenario: TUI tests pass in pivot-tui package
    Tool: Bash
    Steps:
      1. uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto -x
      2. Assert: exit code 0, all tests pass
    Expected Result: TUI tests work in their new home
    Evidence: pytest output

  Scenario: No stale TUI test imports
    Tool: Bash (grep)
    Steps:
      1. grep -rn "from pivot\.tui" packages/pivot-tui/tests/ --include="*.py"
      2. Assert: empty output
      3. grep -rn "from pivot\.engine\.sinks import TuiSink" packages/pivot-tui/tests/ --include="*.py"
      4. Assert: empty output (should be from pivot_tui.sink)
    Expected Result: All test imports updated
    Evidence: grep output
  ```

  **Commit**: YES
  - Message: `test(tui): move TUI tests to pivot-tui package`
  - Files: `packages/pivot-tui/tests/`, deleted test files from `tests/`
  - Pre-commit: `uv run pytest tests/ -n auto -x && uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto -x`

---

- [ ] 9. Tooling updates, import boundary enforcement, and full verification

  **What to do**:
  - **Update basedpyright configuration**:
    - Check if `pyrightconfig.json` (or `pyproject.toml [tool.basedpyright]`) needs updating for workspace structure
    - pivot-tui may need its own pyright config or the root config needs to include `packages/pivot-tui/src`
    - Verify: `uv run basedpyright` passes clean for both packages
  - **Update ruff configuration**:
    - Ensure ruff includes `packages/pivot-tui/src/` and `packages/pivot-tui/tests/` in its scope
    - Check `[tool.ruff]` in root `pyproject.toml` or `ruff.toml` for include/exclude paths
    - Verify: `uv run ruff check .` and `uv run ruff format --check .` pass
  - **Add import boundary check**:
    - Create a script or add a test that verifies zero `import textual` / `from textual` in `src/pivot/`:
      ```python
      # Can be a pytest test in tests/test_import_boundary.py
      import pathlib
      import re

      def test_no_textual_imports_in_core():
          """Core pivot must not import Textual anywhere."""
          root = pathlib.Path("src/pivot")
          violations = []
          for p in root.rglob("*.py"):
              text = p.read_text("utf-8")
              if re.search(r"\bfrom\s+textual\b|\bimport\s+textual\b", text):
                  violations.append(str(p))
          assert not violations, f"Textual imports found in core: {violations}"
      ```
  - **Add core-only import smoke test**:
    - Add a test that imports all core CLI modules to verify they load without pivot-tui:
      ```python
      def test_core_imports_without_textual():
          """All core modules must be importable without Textual."""
          import pivot
          import pivot.cli
          import pivot.engine
          import pivot.types
          # These should not trigger Textual imports
      ```
      Note: This test is meaningful when run in a core-only environment. In the workspace where pivot-tui is installed, it still verifies no top-level Textual imports in core.
  - **Run full quality suite**:
    - `uv run ruff format . && uv run ruff check .`
    - `uv run basedpyright`
    - `uv run pytest tests/ -n auto`
    - `uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto`
  - **Final forbidden-import scan**:
    - `grep -rn "from textual\|import textual" src/pivot/ --include="*.py"` → must be empty
    - `grep -rn "from pivot\.tui" src/pivot/ --include="*.py"` → must be empty
    - `grep -rn "from pivot\.tui" packages/pivot-tui/ --include="*.py"` → must be empty

  **Must NOT do**:
  - Do not fix any pre-existing test failures unrelated to the extraction
  - Do not add new features or change behavior
  - Do not make formatting changes beyond what ruff autoformats

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Final verification across entire workspace, tooling config, potential debugging if issues found
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final, after Tasks 7 and 8)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 7, 8

  **References**:

  **Pattern References**:
  - `pyproject.toml` (root) — basedpyright and ruff config sections
  - `pyrightconfig.json` — If it exists, may need updating for workspace
  - `.ruff.toml` or `ruff.toml` — If it exists, may need path updates

  **Test References**:
  - All test commands listed in Definition of Done section

  **Acceptance Criteria**:

  - [ ] `uv run ruff format --check .` → clean (exit code 0)
  - [ ] `uv run ruff check .` → clean (exit code 0)
  - [ ] `uv run basedpyright` → clean (exit code 0)
  - [ ] `uv run pytest tests/ -n auto` → all core tests pass
  - [ ] `uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto` → all TUI tests pass
  - [ ] Zero `import textual` / `from textual` in `src/pivot/`
  - [ ] Zero `from pivot.tui` in entire codebase
  - [ ] Import boundary test exists and passes
  - [ ] `uv run pivot --help` works

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Full quality suite passes
    Tool: Bash
    Steps:
      1. uv run ruff format --check .
      2. Assert: exit code 0
      3. uv run ruff check .
      4. Assert: exit code 0
      5. uv run basedpyright
      6. Assert: exit code 0
    Expected Result: All quality checks pass
    Evidence: Command outputs

  Scenario: All tests pass across both packages
    Tool: Bash
    Steps:
      1. uv run pytest tests/ -n auto
      2. Assert: exit code 0, all tests pass
      3. uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto
      4. Assert: exit code 0, all tests pass
    Expected Result: Full test coverage maintained
    Evidence: pytest output

  Scenario: Import boundary enforced
    Tool: Bash
    Steps:
      1. grep -rn "from textual\|import textual" src/pivot/ --include="*.py"
      2. Assert: empty output, exit code 1
      3. grep -rn "from pivot\.tui" src/pivot/ --include="*.py"
      4. Assert: empty output, exit code 1
      5. grep -rn "from pivot\.tui" packages/pivot-tui/ --include="*.py"
      6. Assert: empty output, exit code 1
    Expected Result: Zero boundary violations
    Evidence: grep output

  Scenario: CLI works end-to-end
    Tool: Bash
    Steps:
      1. uv run pivot --help
      2. Assert: exit code 0
      3. uv run pivot repro --help
      4. Assert: exit code 0
      5. uv run pivot run --help
      6. Assert: exit code 0
      7. uv run pivot status --help
      8. Assert: exit code 0
      9. uv run pivot data --help
      10. Assert: exit code 0
    Expected Result: All CLI commands work
    Evidence: Command outputs
  ```

  **Commit**: YES
  - Message: `chore: add import boundary tests and finalize workspace tooling`
  - Files: `tests/test_import_boundary.py`, `pyproject.toml` (tooling updates), any config files
  - Pre-commit: `uv run ruff format . && uv run ruff check . && uv run basedpyright && uv run pytest tests/ -n auto`

---

## Commit Strategy

| After Task | Message | Key Files | Verification |
|------------|---------|-----------|--------------|
| 1 | `refactor(tui): move console.py to pivot.cli.console` | console.py, CLI imports | `uv run pytest tests/ -n auto -x` |
| 2 | `refactor: move parse_stage_name to pivot.types` | types.py, _run_common.py | `uv run pytest tests/ -n auto -x` |
| 3 | `build: set up uv workspace with pivot-tui package skeleton` | pyproject.toml, packages/ | `uv sync --all-packages --active` |
| 4 | `refactor(tui): move all Textual TUI code to pivot-tui package` | ~16 files moved (incl. diff.py) | `uv run ruff check packages/pivot-tui/` |
| 5+6 | `refactor(tui): extract TuiSink to pivot-tui, update CLI imports` | sink.py, CLI files (incl. data.py) | `uv run pivot --help` |
| 7 | `refactor(tui): remove old pivot.tui, drop textual from core deps` | old tui/ deleted, pyproject.toml | `uv sync && uv run python -c "import pivot"` |
| 8 | `test(tui): move TUI tests to pivot-tui package` | test files | Both test suites pass |
| 9 | `chore: add import boundary tests, finalize workspace tooling` | tooling configs, boundary test | Full quality suite |

---

## Success Criteria

### Verification Commands
```bash
# Workspace resolves
uv sync --all-packages --active

# Quality
uv run ruff format --check .
uv run ruff check .
uv run basedpyright

# Tests
uv run pytest tests/ -n auto          # Core tests
uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto  # TUI tests

# Import boundary
grep -rn "from textual\|import textual" src/pivot/ --include="*.py"  # Expected: empty
grep -rn "from pivot\.tui" src/ packages/ --include="*.py"           # Expected: empty

# CLI works
uv run pivot --help
uv run pivot repro --help
uv run pivot run --help

# Core works without Textual (import check)
uv run python -c "import pivot; import pivot.cli; import pivot.engine; print('ok')"
```

### Final Checklist
- [ ] All "Must Have" present (one-way dependency, Textual confined, CLI works without pivot-tui)
- [ ] All "Must NOT Have" absent (no new abstractions, no RPC changes, no compat shims)
- [ ] All tests pass (both packages)
- [ ] Import boundary enforced (zero Textual in core)
- [ ] `textual` removed from core dependencies
- [ ] `pivot[tui]` optional extra available
