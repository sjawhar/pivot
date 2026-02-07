# Learnings: TUI Package Extraction

## Conventions & Patterns

(To be populated by agents)

## Code Structure Insights

(To be populated by agents)

## Task 1: uv Workspace Setup

### Completed
- Created `packages/pivot-tui/` directory structure with:
  - `src/pivot_tui/__init__.py` (minimal placeholder)
  - `tests/__init__.py` (empty)
  - `pyproject.toml` with correct metadata
- Updated root `pyproject.toml`:
  - Added `[tool.uv.workspace]` with `members = ["packages/*"]`
  - Added `[tool.uv.sources]` with both `pivot` and `pivot-tui` as workspace members
  - Added `tui = ["pivot-tui"]` to optional dependencies
- Verified workspace resolves: `uv sync --all-packages --active` succeeds
- Verified imports work: `uv run --package pivot-tui python -c "import pivot_tui; print('ok')"` → ok
- Verified core package still works: `uv run pivot --help` succeeds

### Key Learnings
- uv workspace requires ALL workspace members to be declared in `[tool.uv.sources]`
- Both `pivot` (root) and `pivot-tui` (new package) needed workspace source declarations
- hatchling build backend works fine for workspace packages
- Textual dependency stays in core `pivot` package for now (removed in Task 7)

### Pre-existing Issues Noted
- conftest.py imports `from pivot.cli import console` but console is not exported from `__init__.py`
- This is a pre-existing issue unrelated to workspace setup
- Tests fail due to this import error, not due to workspace changes

## Task 1: Move parse_stage_name to pivot.types

**Completed**: parse_stage_name successfully moved from src/pivot/tui/types.py to src/pivot/types.py

### Key Findings

1. **Function Location**: parse_stage_name is a pure string utility (5 lines) that parses stage names with optional "@variant" suffix
   - Returns tuple[str, str]: (base_name, variant) or (name, "")
   - No dependencies on TUI-specific code

2. **Critical Import Site**: src/pivot/cli/_run_common.py:84
   - Used in sort_for_display() which runs BEFORE the TUI check in repro.py
   - Non-TUI runs would crash if function stayed in tui/types.py after pivot-tui extraction
   - This was the primary motivation for the move

3. **Backward Compatibility**: Re-exported from pivot.tui.types
   - Added: `from pivot.types import parse_stage_name` to tui/types.py
   - Allows internal TUI code (StageInfo.__post_init__) to continue using it
   - Prevents breaking changes until Task 4 (full tui package extraction)

4. **Verification Results**:
   - ✅ Function works at new location: `parse_stage_name('train@gpu')` → `('train', 'gpu')`
   - ✅ Function works without variant: `parse_stage_name('simple')` → `('simple', '')`
   - ✅ _run_common.py imports from pivot.types (not tui)
   - ✅ Backward compat works: `from pivot.tui.types import parse_stage_name` still succeeds
   - ✅ Ruff check passes (no style issues)
   - ⚠️ pytest has pre-existing environment issue (pivot.tui.console missing) - not related to this change

### Pattern: String Utilities in pivot.types

parse_stage_name is a good candidate for pivot.types because:
- Pure function (no side effects)
- No TUI dependencies
- Used by non-TUI code paths
- Shared utility across multiple modules

Similar candidates for future moves: any string parsing/formatting utilities currently in tui/types.py

### Import Pattern Used

```python
# In src/pivot/types.py (destination)
def parse_stage_name(name: str) -> tuple[str, str]:
    """Parse stage name into (base_name, variant). Returns (name, '') if no @."""
    if "@" in name:
        base, variant = name.split("@", 1)
        return (base, variant)
    return (name, "")

# In src/pivot/tui/types.py (source - re-export for backward compat)
from pivot.types import OutputChange, StageExplanation, StageStatus, parse_stage_name

# In src/pivot/cli/_run_common.py (consumer - updated import)
from pivot.types import parse_stage_name
```

This pattern ensures:
1. Function is available at its logical home (pivot.types)
2. Existing TUI code continues to work without changes
3. Non-TUI code no longer depends on tui package

### Verification Results
All expected outcomes verified:
- ✓ `packages/pivot-tui/pyproject.toml` exists with correct metadata
- ✓ `packages/pivot-tui/src/pivot_tui/__init__.py` exists
- ✓ Root `pyproject.toml` has `[tool.uv.workspace]` section
- ✓ `uv sync --all-packages --active` succeeds (exit code 0)
- ✓ `uv run --package pivot-tui python -c "import pivot_tui; print('ok')"` → prints `ok`
- ✓ `uv run pivot --help` still works

### Commit
Created commit: `build: set up uv workspace with pivot-tui package skeleton`
- Includes all workspace configuration changes
- Includes pivot-tui package skeleton
- Ready for next task (Task 2: Move TUI code)

## Console Module Relocation (Wave 1)

**Task:** Move `console.py` from `src/pivot/tui/` to `src/pivot/cli/`

**Completed:** 2026-02-07

### Key Findings

1. **File was already at new location in git** - The file `src/pivot/cli/console.py` was already tracked in git, so the move was a matter of updating imports and deleting the old file from the filesystem.

2. **Import sites found via LSP** - Used `lsp_find_references` on the `Console` class to identify all usage sites:
   - `src/pivot/cli/repro.py` (2 locations: lines 44 and 231)
   - `src/pivot/cli/run.py` (1 location: line 264)
   - `src/pivot/cli/status.py` (1 location: line 15)
   - `tests/conftest.py` (1 location: line 23)
   - `tests/cli/test_cli.py` (1 location: line 12)
   - `tests/tui/test_console.py` (1 location: line 6)

3. **Indentation pitfalls** - When editing imports in function bodies (repro.py line 230, run.py line 263), the indentation must match the surrounding code exactly. Initial edits added extra spaces causing IndentationError.

4. **Ruff auto-fix** - Used `uv run ruff check src/ --fix` to automatically fix import ordering issues in 4 files.

5. **Test coverage** - All 3499 tests pass after the refactor, confirming no behavioral changes.

### Changes Made

- Deleted `src/pivot/tui/console.py` (old location)
- Updated `src/pivot/tui/__init__.py` to remove console export and fix relative imports
- Updated all 6 import sites to use `from pivot.cli import console`
- Fixed import ordering with ruff

### Verification

✓ Import works: `from pivot.cli.console import Console`
✓ No references to `pivot.tui.console` remain
✓ All 3499 tests pass
✓ Ruff checks clean

## Task 4: Move All Textual TUI Code to pivot-tui Package

**Completed**: 2026-02-07

### Files Moved
- 19 Python files + 2 TCSS files moved from `src/pivot/tui/` to `packages/pivot-tui/src/pivot_tui/`
- Directory structure preserved: widgets/, screens/, styles/ subdirectories
- `src/pivot/tui/__init__.py` kept as empty stub (just `from __future__ import annotations`)

### Import Rewriting
- 28 `from pivot.tui.*` import statements updated to `from pivot_tui.*`
- Used `sed -i 's/from pivot\.tui\./from pivot_tui./g'` for bulk replacement across all .py files
- Separate sed for `from pivot.tui import` (no trailing dot) in run.py
- Ruff auto-fix resolved 4 import ordering issues (pivot_tui is first-party, pivot is third-party from pivot-tui's perspective)

### Key Pattern: Import Ordering with Workspace Packages
- When `pivot_tui` imports from both `pivot` (workspace dependency) and `pivot_tui` (first-party), ruff expects:
  - `pivot` imports grouped with third-party (since it's a separate package)
  - `pivot_tui` imports grouped with first-party
- This caused I001 errors after the bulk rename; `ruff check --fix` resolved them

### __init__.py Pattern
- Preserved same export pattern as original: exports `diff` and `diff_panels`, NOT `run`
- `run` is not imported at package level to avoid circular imports with executor

### Verification Results
- ✅ `import pivot_tui` → ok
- ✅ `from pivot_tui.types import StageDataProvider` → ok
- ✅ `from pivot_tui.rpc_client import send_run_command` → ok
- ✅ All sub-modules importable (widgets, screens, stats)
- ✅ Cross-package imports work (pivot_tui.types + pivot.types together)
- ✅ No `from pivot.tui` references remain in packages/pivot-tui/src/
- ✅ TCSS files moved correctly
- ✅ `ruff check packages/pivot-tui/src/` → clean
- ✅ `ruff format` → no changes needed

## Task 5: CLI Lazy Imports from pivot_tui

**Completed**: 2026-02-07

### Objective
Update CLI modules to use lazy imports from `pivot_tui` with graceful error handling when pivot-tui is not installed. This establishes a one-way dependency boundary: core pivot → pivot_tui (not the reverse).

### Changes Made

1. **src/pivot/cli/run.py** (line 87)
   - Changed: `from pivot.tui import run as tui_run` (top-level in function)
   - To: Lazy import with try/except inside `_run_with_tui()` function
   - Error handling: Raises `click.UsageError` with installation instructions

2. **src/pivot/cli/repro.py** (lines 378 and 612)
   - Changed: Two occurrences of `from pivot.tui import run as tui_run`
   - To: Lazy imports with try/except in `_run_watch_mode()` and `_run_oneshot_mode()` functions
   - Error handling: Raises `click.UsageError` with installation instructions

3. **src/pivot/cli/data.py** (line 129)
   - Changed: `from pivot.tui import diff as data_tui` (in else block)
   - To: Lazy import with try/except in the TUI code path
   - Error handling: Raises `click.UsageError` with installation instructions

4. **src/pivot/cli/_run_common.py** (line 26)
   - Changed: `from pivot.tui.run import MessagePoster` (TYPE_CHECKING block)
   - To: `from pivot_tui.run import MessagePoster` (TYPE_CHECKING block)
   - Already lazy (TYPE_CHECKING only) - just updated package name

### Verification Results

✅ **No remaining pivot.tui imports in CLI**
```bash
grep -rn "from pivot\.tui" src/pivot/cli/ --include="*.py"  # No output
grep -rn "import pivot\.tui" src/pivot/cli/ --include="*.py"  # No output
```

✅ **CLI help commands work without pivot-tui installed**
- `uv run pivot --help` → exit code 0
- `uv run pivot repro --help` → exit code 0
- `uv run pivot run --help` → exit code 0

✅ **Ruff checks pass**
- Fixed import sorting in _run_common.py (I001)
- Fixed exception chaining with `from err` (B904)
- All checks pass: `uv run ruff check src/pivot/cli/` → clean

### Pattern: Lazy Imports with Error Handling

```python
# Inside function body (not at module level)
try:
    from pivot_tui import run as tui_run
except ImportError as err:
    raise click.UsageError(
        "The TUI requires the 'pivot-tui' package. "
        "Install it with: uv pip install pivot-tui"
    ) from err
```

This pattern ensures:
1. Core CLI loads without pivot-tui dependency
2. TUI features fail gracefully with clear error message
3. Users know exactly how to fix the issue
4. Exception chaining preserves original ImportError for debugging

### Dependency Boundary Established

**Before**: Core pivot → pivot.tui (circular risk)
**After**: Core pivot → pivot_tui (one-way, lazy)

The CLI can now be used without pivot-tui installed. Only when `--tui` flag is used does the import happen, and it fails with a helpful error message if the package is missing.

### Files Modified
- src/pivot/cli/run.py
- src/pivot/cli/repro.py
- src/pivot/cli/data.py
- src/pivot/cli/_run_common.py

### Commit
YES - Message: `refactor(cli): update TUI imports to lazy-load from pivot-tui package`

## Task 5: Extract TuiSink from engine/sinks.py to pivot-tui

### Completed
- ✓ Created `packages/pivot-tui/src/pivot_tui/sink.py` with TuiSink class
- ✓ Moved helper functions: `_make_started_message`, `_make_completed_message`, `_make_log_message`
- ✓ Removed TuiSink from `src/pivot/engine/sinks.py`
- ✓ Cleaned up unused imports from engine/sinks.py (TuiLogMessage, TuiMessageType, TuiStatusMessage, time, anyio.to_thread)
- ✓ Updated `src/pivot/cli/_run_common.py` to use lazy import: `from pivot_tui.sink import TuiSink`
- ✓ Updated test imports in:
  - `tests/engine/test_tui_sink.py`: Changed from `pivot.engine.sinks` to `pivot_tui.sink`
  - `tests/tui/test_tui_integration.py`: Changed from `pivot.tui.run` to `pivot_tui.run`
  - `tests/cli/test_cli_run_common.py`: Added import of TuiSink from `pivot_tui.sink`

### Key Decisions
1. **Lazy import in _run_common.py**: TuiSink is only imported when `tui=True`, avoiding circular imports
2. **No export from pivot_tui/__init__.py**: Avoided circular import with run.py by not exporting TuiSink from __init__.py
3. **TYPE_CHECKING for StageCompleted**: Moved StageCompleted to TYPE_CHECKING block in sink.py to avoid runtime import

### Circular Import Pattern
- `pivot_tui/__init__.py` imports from `pivot_tui.run`
- `pivot_tui.run` imports from `pivot_tui` (via `from pivot_tui import diff_panels, rpc_client`)
- Solution: Don't export sink from __init__.py; use direct import `from pivot_tui.sink import TuiSink`

### Test Results
All 10 tests pass:
- 8 tests in test_tui_sink.py
- 1 test in test_cli_run_common.py (configure_output_sink_tui_mode)
- 1 test in test_tui_integration.py

### Ruff Status
All files pass ruff checks (no errors or warnings)

## Task 7: Remove Old TUI Directory & Drop Textual from Core

**Completed:** 2026-02-07

### Changes Made
1. **Removed** `src/pivot/tui/` directory entirely (contained only `__init__.py` and pycache)
2. **Updated** root `pyproject.toml`:
   - Removed `textual>=7.0` from `[project.dependencies]`
   - Added `rich>=13.0` to core dependencies (needed by `pivot.cli.console`)
   - Kept `tqdm>=4.66` in core (needed by `pivot.cli.console`)
   - Verified `tui = ["pivot-tui"]` in `[project.optional-dependencies]`

### Verification
- ✅ `uv sync --active` succeeds (core only, no TUI)
- ✅ `uv run python -c "import pivot; import pivot.cli; import pivot.engine; print('ok')"` → prints `ok`
- ✅ `src/pivot/tui/` directory no longer exists
- ✅ `textual` not in root `pyproject.toml` dependencies
- ✅ `rich` and `tqdm` remain in core dependencies

### Key Insight
The old `src/pivot/tui/` was a stub directory with only `__init__.py`. All actual TUI code had already been moved to `packages/pivot-tui/` in Tasks 1-6. This task completed the decoupling by removing the empty stub and dropping the Textual dependency from core.

### Dependency Structure
- **Core** (`pivot`): `rich`, `tqdm` (for CLI console output)
- **Optional** (`pivot[tui]`): `pivot-tui` (which brings in `textual`)
- This allows users to install core without TUI overhead

## Task 7: Move TUI Tests to pivot-tui Package

**Completed**: 2026-02-07

### Objective
Move all TUI-specific test files from `tests/` to `packages/pivot-tui/tests/` and update imports to use `pivot_tui` instead of `pivot.tui`.

### Files Moved
- **From tests/tui/ → packages/pivot-tui/tests/**
  - test_console.py
  - test_diff.py
  - test_diff_panels.py
  - test_history.py
  - test_log_search_e2e.py
  - test_rpc_client.py
  - test_run.py
  - test_stats.py
  - widgets/test_footer.py
  - widgets/test_log_level_detection.py
  - widgets/test_logs.py
  - widgets/test_status.py

- **From tests/engine/test_tui_sink.py → packages/pivot-tui/tests/**
  - test_tui_sink.py

- **From tests/integration/test_tui_force_rerun.py → packages/pivot-tui/tests/**
  - test_tui_force_rerun.py

- **From tests/cli/test_watch.py → packages/pivot-tui/tests/**
  - test_watch.py (extracted TUI-specific tests)

- **From tests/cli/test_cli_run_common.py → packages/pivot-tui/tests/**
  - test_cli_run_common.py (extracted test_configure_output_sink_tui_mode)

### Import Updates
1. **pivot.tui → pivot_tui**: Updated all `from pivot.tui import X` and `from pivot.tui.Y import Z` statements
2. **pivot.engine.sinks.TuiSink → pivot_tui.sink.TuiSink**: Updated TuiSink imports
3. **Mocker patches**: Updated all `mocker.patch("pivot.tui.X")` to `mocker.patch("pivot_tui.X")`

### Supporting Files Copied
- `tests/conftest.py` → `packages/pivot-tui/tests/conftest.py` (provides send_rpc fixture)
- `tests/helpers.py` → `packages/pivot-tui/tests/helpers.py` (test utilities)

### Test Results
- ✅ Core tests: 3133 passed, 4 skipped, 6 xfailed
- ✅ TUI tests: 366 passed, 1 rerun
- ✅ No TUI tests remain in root test suite
- ✅ All imports updated correctly

### Key Learnings

1. **Test File Organization**: TUI tests are now fully isolated in the pivot-tui package, making it clear which tests depend on TUI functionality.

2. **Conftest and Helpers**: Test support files (conftest.py, helpers.py) need to be copied to the new test location since pytest discovers them relative to the test directory.

3. **Mocker Patches**: When moving tests between packages, all mocker.patch() calls that reference the old module path must be updated. This includes:
   - Direct module patches: `"pivot.tui.module.function"`
   - Nested patches: `"pivot.tui.module.submodule.function"`

4. **Test Isolation**: The separation enables:
   - Running core tests without TUI dependencies: `uv run pytest tests/ -n auto`
   - Running TUI tests in isolation: `uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto`
   - Clear dependency boundaries between core and TUI functionality

### Files Deleted from Root
- `tests/tui/` (entire directory)
- `tests/engine/test_tui_sink.py`
- `tests/integration/test_tui_force_rerun.py`
- `tests/cli/test_watch.py` (after extracting TUI tests)
- `test_configure_output_sink_tui_mode` from `tests/cli/test_cli_run_common.py`

### Verification Commands
```bash
# Core tests pass without TUI tests
uv run pytest tests/ -n auto -x

# TUI tests pass in pivot-tui package
uv run --package pivot-tui pytest packages/pivot-tui/tests/ -n auto -x

# No stale TUI test imports
grep -rn "from pivot\.tui" packages/pivot-tui/tests/ --include="*.py"  # Should be empty
grep -rn "from pivot\.engine\.sinks import TuiSink" packages/pivot-tui/tests/ --include="*.py"  # Should be empty
```

### Commit
YES - Message: `test(tui): move TUI tests to pivot-tui package`
Files: `packages/pivot-tui/tests/`, deleted test files from `tests/`
Pre-commit: Both test suites pass
