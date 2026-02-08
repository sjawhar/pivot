# Finish Monorepo Repackaging: pivot as Workspace Member

## TL;DR

> **Quick Summary**: Complete the pivot monorepo restructuring by creating `packages/pivot/pyproject.toml`, slimming the root to workspace-only config, switching pivot-tui to uv_build, and updating all tool path references.
> 
> **Deliverables**:
> - `packages/pivot/pyproject.toml` — full package definition with deps, scripts, and dev extra
> - Updated root `pyproject.toml` — workspace-only + tool config (no project metadata)
> - Updated `packages/pivot-tui/pyproject.toml` — switched from hatchling to uv_build, dev extra added
> - All tool configs point to new `packages/` paths
> - Updated `AGENTS.md` with correct paths
> - Regenerated `uv.lock`
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: NO — sequential (each step depends on prior)
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6

---

## Context

### Original Request
Finish monorepo repackaging of pivot. Source code and tests already moved from `src/pivot/` → `packages/pivot/src/pivot/` and `tests/` → `packages/pivot/tests/`. Need to create the per-package pyproject.toml, update root config, fix all path references, and get everything building/testing again.

### Interview Summary
**Key Discussions**:
- Build backend: Both packages use `uv_build` (pivot-tui needs switch from hatchling)
- Dev deps: Each package declares dev deps as `[project.optional-dependencies] dev = [...]`, root installs both with dev extras
- Tool config: Centralized in root `pyproject.toml` with updated paths pointing into `packages/`

**Research Findings**:
- Root `pyproject.toml` already has workspace config (`members = ["packages/*"]`, sources for both packages)
- `uv.lock` currently shows `pivot source = { editable = "." }` — will auto-fix after `uv lock`
- basedpyright has 4 execution environments all referencing old `src/` and `tests/` paths
- mkdocs handler has `paths: [src]` for Python path resolution
- `packages/pivot-tui/tests/conftest.py` is a near-copy of `packages/pivot/tests/conftest.py` — both import pivot internals for fixture setup
- pivot-tui depends on pivot; pivot lazily imports pivot_tui (inside function bodies, behind optional dep)

### Metis Review
**Identified Gaps** (addressed):
- Tool path misalignment risk — addressed by including explicit path verification in acceptance criteria
- Hatchling→uv_build swap may lose metadata — addressed by task to verify equivalent config
- Shared conftest coupling — addressed by running both test suites as acceptance criteria
- Lockfile staleness — addressed by explicit `uv lock` task with source path verification

---

## Work Objectives

### Core Objective
Make `packages/pivot/` a proper uv workspace member with its own pyproject.toml, slim the root to workspace-only config, and ensure all tooling works with the new directory layout.

### Concrete Deliverables
- `packages/pivot/pyproject.toml` (new file)
- `pyproject.toml` (edited — workspace root only)
- `packages/pivot-tui/pyproject.toml` (edited — uv_build + dev extra)
- `mkdocs.yml` (edited — updated paths)
- `AGENTS.md` (edited — updated paths)
- `uv.lock` (regenerated)

### Definition of Done
- [x] `uv sync --active` succeeds
- [x] `uv run pytest packages/pivot/tests -n auto` passes
- [x] `uv run pytest packages/pivot-tui/tests` passes
- [x] `uv run ruff check .` passes
- [x] `uv run ruff format --check .` passes
- [x] `uv run basedpyright` passes
- [x] `pivot` CLI entry point works: `uv run pivot --help`

### Must Have
- `packages/pivot/pyproject.toml` with all runtime deps, optional deps, scripts, build system
- Dev extras on both packages
- Root installs `pivot[dev]` and `pivot-tui[dev]`
- All tool configs reference `packages/` paths

### Must NOT Have (Guardrails)
- Do NOT add new dependencies that don't already exist
- Do NOT modify runtime behavior or API surface
- Do NOT move any more files — the file moves are done
- Do NOT change Python import paths (`from pivot.xxx` stays the same)
- Do NOT create thin wrappers or compatibility shims
- Do NOT split tool config into per-package files unless a tool genuinely can't handle root-level paths

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: YES (tests-after — verify existing tests still pass)
- **Framework**: pytest with pytest-xdist

---

## Execution Strategy

### Sequential Execution
Tasks must run sequentially because each builds on the prior:
1. Create packages/pivot/pyproject.toml (needed before root can be slimmed)
2. Slim root pyproject.toml (depends on package pyproject existing)
3. Update pivot-tui pyproject.toml (independent of 1-2, but logically grouped)
4. Update tool configs in root (depends on 2)
5. Update AGENTS.md + mkdocs.yml (depends on 4 for context)
6. Regenerate lockfile + verify (depends on all above)

---

## TODOs

- [x] 1. Create `packages/pivot/pyproject.toml`

  **What to do**:
  - Create `packages/pivot/pyproject.toml` with:
    - `[project]`: name="pivot", version="0.1.0", description, requires-python=">=3.13,<3.14", license, authors
    - `[project.dependencies]`: All current runtime deps from root pyproject.toml (anyio, click, deepmerge, dulwich, filelock, flatten-dict, grandalf, lmdb, loky, networkx, pandas, pathspec, pydantic, pygtrie, pyyaml, rich, ruamel.yaml, tabulate, tqdm, watchfiles, xxhash)
    - `[project.optional-dependencies]`: 
      - `dvc = ["dvc>=3.0"]`
      - `matplotlib = ["matplotlib>=3.5"]`
      - `s3 = ["aioboto3>=13.0.0"]`
      - `tui = ["pivot-tui"]`
      - `docs = [...]` (same as current root)
      - `dev = [...]` — NEW: all current dev deps that are specific to testing pivot (pytest, pytest-xdist, pytest-cov, pytest-mock, pytest-asyncio, pytest-rerunfailures, pytest-watcher, pytest-aioboto3, ruff, basedpyright, mypy, pandas-stubs, botocore-stubs, matplotlib-stubs, joblib, joblib-stubs, types-aioboto3[all], plus the optional extras: `pivot[dvc]`, `pivot[matplotlib]`, `pivot[s3]`)
    - `[project.scripts]`: `pivot = "pivot.cli:main"`
    - `[build-system]`: requires=["uv_build>=0.9.18,<0.10.0"], build-backend="uv_build"
  - The `[tool.uv.sources]` in the **root** pyproject already has `pivot-tui = { workspace = true }` — the package-level pyproject.toml for pivot should also have `[tool.uv.sources] pivot-tui = { workspace = true }` so the `tui` optional dep resolves within the workspace.

  **Must NOT do**:
  - Do not add any deps that aren't in the current root pyproject.toml
  - Do not add tool config here (stays in root)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
    - Simple file creation from known content

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential — Task 1
  - **Blocks**: Tasks 2, 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `packages/pivot-tui/pyproject.toml` — Reference for workspace member pyproject.toml structure (though note it currently uses hatchling — the new file should use uv_build)
  
  **Source of Truth**:
  - `pyproject.toml` (root, lines 1-50) — Contains all current project metadata, dependencies, optional-dependencies, and scripts that need to move to the new file
  - `pyproject.toml` (root, lines 55-77) — Current dev dependency-group that will become the `dev` optional extra

  **Acceptance Criteria**:

  - [x] File exists: `packages/pivot/pyproject.toml`
  - [x] Build backend is uv_build: `python -c "import tomllib; print(tomllib.loads(open('packages/pivot/pyproject.toml','rb').read())['build-system']['build-backend'])"` → `uv_build`
  - [x] Has all runtime deps (22 packages): verify count from `[project.dependencies]`
  - [x] Has `dev` optional extra with test deps
  - [x] Has `pivot = "pivot.cli:main"` in `[project.scripts]`
  - [x] Has `[tool.uv.sources] pivot-tui = { workspace = true }`

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify pyproject.toml structure is valid TOML with required sections
    Tool: Bash
    Preconditions: File created at packages/pivot/pyproject.toml
    Steps:
      1. python -c "import tomllib, pathlib; d=tomllib.loads(pathlib.Path('packages/pivot/pyproject.toml').read_bytes()); assert d['project']['name']=='pivot'; assert 'uv_build' in d['build-system']['build-backend']; assert 'pivot.cli:main' in str(d['project']['scripts']); assert 'dev' in d['project']['optional-dependencies']; print('OK')"
    Expected Result: Prints "OK"
    Evidence: Command output captured
  ```

  **Commit**: NO (groups with Task 6)

---

- [x] 2. Slim root `pyproject.toml` to workspace-only

  **What to do**:
  - Remove `[project]` section entirely (name, version, description, readme, requires-python, license, authors, dependencies, optional-dependencies, scripts)
  - Remove `[build-system]` section
  - Remove `[dependency-groups] dev` section
  - Add `[project]` with just `name = "pivot-workspace"`, `version = "0.1.0"`, `requires-python = ">=3.13,<3.14"` (minimal metadata required by uv for workspace root)
  - Add `[dependency-groups] dev` that references both packages with dev extras: `dev = ["pivot[dev]", "pivot-tui[dev]"]`
  - Keep `[tool.uv.workspace]` and `[tool.uv.sources]` as-is
  - Keep ALL `[tool.*]` sections (basedpyright, coverage, pytest, ruff, etc.) — these get updated in Task 4
  - Keep `[tool.semantic_release]` as-is (may need path update later but out of scope)

  **Must NOT do**:
  - Do not remove tool config sections
  - Do not remove workspace config
  - Do not change tool paths yet (Task 4)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential — Task 2
  - **Blocks**: Tasks 4, 6
  - **Blocked By**: Task 1

  **References**:

  **Source of Truth**:
  - `pyproject.toml` (root, lines 1-230) — Current root pyproject.toml. Lines 1-49 are project metadata to remove/replace. Lines 51-53 are build-system to remove. Lines 55-77 are dev dependency-group to replace. Lines 220-225 are workspace config to keep. Lines 80-215 are tool config to keep.

  **Acceptance Criteria**:

  - [x] Root pyproject.toml has `name = "pivot-workspace"` (not `"pivot"`)
  - [x] Root pyproject.toml has NO `[project.dependencies]` (no runtime deps)
  - [x] Root pyproject.toml has NO `[project.scripts]`
  - [x] Root pyproject.toml has NO `[build-system]` with uv_build (workspace roots don't need build-system)
  - [x] Root pyproject.toml has `[dependency-groups] dev` with `pivot[dev]` and `pivot-tui[dev]`
  - [x] Root pyproject.toml still has `[tool.uv.workspace]` and `[tool.uv.sources]`
  - [x] Root pyproject.toml still has all `[tool.*]` sections

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify root is workspace-only
    Tool: Bash
    Preconditions: Root pyproject.toml edited
    Steps:
      1. python -c "
      import tomllib, pathlib
      d = tomllib.loads(pathlib.Path('pyproject.toml').read_bytes())
      assert d['project']['name'] == 'pivot-workspace', f'name={d[\"project\"][\"name\"]}'
      assert 'dependencies' not in d['project'], 'has runtime deps'
      assert 'scripts' not in d['project'], 'has scripts'
      assert 'workspace' in d['tool']['uv'], 'missing workspace config'
      assert 'pivot[dev]' in d['dependency-groups']['dev'], 'missing pivot[dev]'
      assert 'pivot-tui[dev]' in d['dependency-groups']['dev'], 'missing pivot-tui[dev]'
      print('OK')
      "
    Expected Result: Prints "OK"
    Evidence: Command output captured
  ```

  **Commit**: NO (groups with Task 6)

---

- [x] 3. Update `packages/pivot-tui/pyproject.toml`

  **What to do**:
  - Switch build system from hatchling to uv_build:
    - Change `[build-system]` to: `requires = ["uv_build>=0.9.18,<0.10.0"]`, `build-backend = "uv_build"`
    - Remove `[tool.hatch.build.targets.wheel]` section entirely (uv_build auto-discovers `src/` layout)
  - Add `[project.optional-dependencies]` with `dev` extra containing TUI-specific test deps:
    - `dev = ["pytest>=7.0", "pytest-asyncio>=0.24.0", "pytest-mock>=3.15.1"]`
    - Note: textual ships its own test harness, no extra test dep needed
  - Add `[tool.uv.sources]` with `pivot = { workspace = true }` so the `pivot` dependency resolves within workspace
  - Keep existing `[tool.pytest.ini_options]` (it has per-package test config, which is appropriate)

  **Must NOT do**:
  - Do not duplicate tool config that belongs in root
  - Do not add deps that aren't currently used by pivot-tui tests

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential — Task 3
  - **Blocks**: Task 6
  - **Blocked By**: Task 2 (logically — root config should be settled first)

  **References**:

  **Source of Truth**:
  - `packages/pivot-tui/pyproject.toml` (lines 1-27) — Current file to edit. Lines 11-16 are hatchling config to replace with uv_build. All 27 lines visible.

  **Acceptance Criteria**:

  - [x] Build backend is uv_build (not hatchling)
  - [x] No `[tool.hatch.*]` sections remain
  - [x] `dev` optional extra exists
  - [x] `[tool.uv.sources]` has `pivot = { workspace = true }`

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify pivot-tui pyproject updated correctly
    Tool: Bash
    Preconditions: packages/pivot-tui/pyproject.toml edited
    Steps:
      1. python -c "
      import tomllib, pathlib
      d = tomllib.loads(pathlib.Path('packages/pivot-tui/pyproject.toml').read_bytes())
      assert 'uv_build' in d['build-system']['build-backend'], 'wrong backend'
      assert 'hatch' not in str(d.get('tool', {})), 'hatch config remains'
      assert 'dev' in d['project']['optional-dependencies'], 'missing dev extra'
      assert d['tool']['uv']['sources']['pivot'] == {'workspace': True}, 'missing workspace source'
      print('OK')
      "
    Expected Result: Prints "OK"
    Evidence: Command output captured
  ```

  **Commit**: NO (groups with Task 6)

---

- [x] 4. Update tool config paths in root `pyproject.toml`

  **What to do**:
  - **basedpyright execution environments** (4 sections):
    - Change `root = "src"` → `root = "packages/pivot/src"` (with `extraPaths = ["packages/pivot-tui/src"]`)
    - Change `root = "tests"` → `root = "packages/pivot/tests"` (with `extraPaths = [".", "packages/pivot-tui/src"]`)
    - Keep `root = "packages/pivot-tui/src"` as-is
    - Keep `root = "packages/pivot-tui/tests"` as-is (but update extraPaths if needed)
  - **ruff `src`**: Change `["src", "tests", "packages/pivot-tui/src", "packages/pivot-tui/tests"]` → `["packages/pivot/src", "packages/pivot/tests", "packages/pivot-tui/src", "packages/pivot-tui/tests"]`
  - **ruff per-file-ignores**: Change `"tests/**/*.py"` → `"packages/pivot/tests/**/*.py"`
  - **pytest testpaths**: Change `["tests"]` → `["packages/pivot/tests", "packages/pivot-tui/tests"]`
  - **coverage source**: Change `["src/pivot"]` → `["packages/pivot/src/pivot"]`
  - **coverage omit**: Change `["tests/*", ...]` → `["packages/pivot/tests/*", ...]`

  **Must NOT do**:
  - Do not change tool behavior/rules — only paths
  - Do not add new lint rules or exclusions

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential — Task 4
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Task 2

  **References**:

  **Source of Truth**:
  - `pyproject.toml` (root, lines 80-213) — All tool config sections. Lines 80-128 are basedpyright. Lines 129-151 are coverage. Lines 153-165 are pytest. Lines 172-213 are ruff.
  
  **Current values to change**:
  - Line 99: `root = "src"` → `root = "packages/pivot/src"`
  - Line 100: `extraPaths = ["packages/pivot-tui/src"]` — keep as-is
  - Line 103: `root = "tests"` → `root = "packages/pivot/tests"`
  - Line 104: `extraPaths = [".", "packages/pivot-tui/src"]` — keep as-is
  - Line 130: `source = ["src/pivot"]` → `source = ["packages/pivot/src/pivot"]`
  - Line 134: `omit = ["tests/*", ...]` → `omit = ["packages/pivot/tests/*", ...]`
  - Line 154: `testpaths = ["tests"]` → `testpaths = ["packages/pivot/tests", "packages/pivot-tui/tests"]`
  - Line 175: `src = ["src", "tests", ...]` → `src = ["packages/pivot/src", "packages/pivot/tests", ...]`
  - Line 198: `"tests/**/*.py"` → `"packages/pivot/tests/**/*.py"`

  **Acceptance Criteria**:

  - [x] No references to bare `"src"` or `"tests"` in tool config (all prefixed with `packages/`)
  - [x] basedpyright execution environments point to correct roots
  - [x] ruff src includes all 4 paths under `packages/`
  - [x] pytest testpaths includes both test directories
  - [x] coverage source points to `packages/pivot/src/pivot`

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify no stale path references remain
    Tool: Bash
    Preconditions: Root pyproject.toml tool config updated
    Steps:
      1. python -c "
      import tomllib, pathlib
      d = tomllib.loads(pathlib.Path('pyproject.toml').read_bytes())
      # Check basedpyright
      envs = d['tool']['basedpyright']['executionEnvironments']
      roots = [e['root'] for e in envs]
      assert 'src' not in roots, f'stale src root: {roots}'
      assert 'tests' not in roots, f'stale tests root: {roots}'
      assert 'packages/pivot/src' in roots, f'missing pivot src: {roots}'
      assert 'packages/pivot/tests' in roots, f'missing pivot tests: {roots}'
      # Check ruff
      ruff_src = d['tool']['ruff']['src']
      assert 'src' not in ruff_src, f'stale src in ruff: {ruff_src}'
      assert 'packages/pivot/src' in ruff_src
      # Check pytest
      tp = d['tool']['pytest']['ini_options']['testpaths']
      assert 'tests' not in tp, f'stale tests in pytest: {tp}'
      assert 'packages/pivot/tests' in tp
      # Check coverage
      cs = d['tool']['coverage']['run']['source']
      assert 'src/pivot' not in cs, f'stale src/pivot in coverage: {cs}'
      assert 'packages/pivot/src/pivot' in cs
      print('OK')
      "
    Expected Result: Prints "OK"
    Evidence: Command output captured
  ```

  **Commit**: NO (groups with Task 6)

---

- [x] 5. Update `AGENTS.md` and `mkdocs.yml` path references

  **What to do**:
  - **AGENTS.md** (root):
    - Update project structure tree: `src/pivot/` → `packages/pivot/src/pivot/`
    - Update key entry points table: all `src/pivot/...` → `packages/pivot/src/pivot/...`
    - Fix key entry points: `state_db.py` → `state.py` (the file is actually `packages/pivot/src/pivot/storage/state.py`, not `state_db.py`)
    - Update `WorkerStageInfo` reference: `src/pivot/executor/worker.py` → `packages/pivot/src/pivot/executor/worker.py`
    - Update development commands: `uv run pytest tests/ -n auto` → `uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto`
    - Remove `tui/` from the project structure tree (TUI is separate package now)
    - Update coverage source reference if present
  - **mkdocs.yml**:
    - Line 47: Change `paths: [src]` → `paths: [packages/pivot/src]` in mkdocstrings handler config
  - **Per-package AGENTS.md files** (`packages/pivot/src/pivot/cli/AGENTS.md`, `packages/pivot/src/pivot/executor/AGENTS.md`, `packages/pivot/src/pivot/storage/AGENTS.md`, `packages/pivot/tests/AGENTS.md`): Scan for any `src/pivot/` path references that should be updated. These files were moved (renamed) but their contents may reference the old paths.

  **Must NOT do**:
  - Do not change the substance/meaning of documentation — only paths
  - Do not create new documentation files

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential — Task 5
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:

  **Source of Truth**:
  - `AGENTS.md` (root, lines 1-198) — Contains project structure tree (lines 11-28), key entry points (lines 32-38), WorkerStageInfo reference (line 70), development commands (lines 191-194)
  - `mkdocs.yml` (line 47) — `paths: [src]` in mkdocstrings handler
  - `packages/pivot/src/pivot/cli/AGENTS.md` — May contain path references
  - `packages/pivot/src/pivot/executor/AGENTS.md` — May contain path references
  - `packages/pivot/src/pivot/storage/AGENTS.md` — May contain path references
  - `packages/pivot/tests/AGENTS.md` — May contain path references

  **Acceptance Criteria**:

  - [x] Root `AGENTS.md` has no bare `src/pivot/` references (should be `packages/pivot/src/pivot/`)
  - [x] Root `AGENTS.md` development commands reference correct test paths
  - [x] `mkdocs.yml` paths updated to `packages/pivot/src`
  - [x] No `tui/` entry in the pivot package project structure tree (it's a separate package)

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Verify no stale src/pivot paths in root AGENTS.md
    Tool: Bash
    Preconditions: AGENTS.md updated
    Steps:
      1. grep -n "src/pivot/" AGENTS.md | grep -v "packages/pivot/src/pivot/" | grep -v "packages/pivot-tui/src/"
      2. Assert: no output (exit code 1 = no matches = correct)
    Expected Result: No stale src/pivot/ references found
    Evidence: grep output captured

  Scenario: Verify mkdocs paths updated
    Tool: Bash
    Preconditions: mkdocs.yml updated
    Steps:
      1. python -c "
      import yaml
      with open('mkdocs.yml') as f:
          d = yaml.safe_load(f)
      handler = d['plugins'][3]['mkdocstrings']['handlers']['python']
      assert handler['paths'] == ['packages/pivot/src'], f'wrong paths: {handler[\"paths\"]}'
      print('OK')
      "
    Expected Result: Prints "OK"
    Evidence: Command output captured
  ```

  **Commit**: NO (groups with Task 6)

---

- [x] 6. Regenerate lockfile and verify everything works

  **What to do**:
  - Run `uv lock` to regenerate `uv.lock` with new package structure
  - Run `uv sync --active` to install both packages in editable mode
  - Run full test suite: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto`
  - Run quality checks: `uv run ruff format --check .`, `uv run ruff check .`, `uv run basedpyright`
  - Verify CLI entry point: `uv run pivot --help`
  - Fix any issues that arise from the restructuring (path mismatches, import errors, etc.)

  **Must NOT do**:
  - Do not change runtime code to fix test failures — the code should work as-is with correct config
  - Do not skip failing tests — fix the root cause
  - Do not add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
    - This task may require debugging and iterating if issues are found

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential — Task 6 (final)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 1-5

  **References**:

  **Documentation References**:
  - `AGENTS.md` (root, lines 189-197) — Development commands and quality check expectations

  **Acceptance Criteria**:

  - [x] `uv lock` succeeds (exit code 0)
  - [x] `uv sync --active` succeeds (exit code 0)
  - [x] `uv.lock` contains `source = { editable = "packages/pivot" }` for pivot (not `editable = "."`)
  - [x] `uv run pivot --help` shows CLI help
  - [x] `uv run pytest packages/pivot/tests -n auto` passes (exit code 0)
  - [x] `uv run pytest packages/pivot-tui/tests` passes (exit code 0)
  - [x] `uv run ruff format --check .` passes
  - [x] `uv run ruff check .` passes
  - [x] `uv run basedpyright` passes

  **Agent-Executed QA Scenarios (MANDATORY):**

  ```
  Scenario: Full workspace verification
    Tool: Bash
    Preconditions: All prior tasks complete
    Steps:
      1. uv lock
      2. Assert: exit code 0
      3. uv sync --active
      4. Assert: exit code 0
      5. grep 'editable = "packages/pivot"' uv.lock
      6. Assert: match found (pivot is workspace member at correct path)
      7. uv run pivot --help
      8. Assert: exit code 0, output contains "pivot"
    Expected Result: Workspace installs correctly, CLI works
    Evidence: Command outputs captured

  Scenario: Test suite passes
    Tool: Bash
    Preconditions: uv sync complete
    Steps:
      1. uv run pytest packages/pivot/tests -n auto --timeout=120
      2. Assert: exit code 0
      3. uv run pytest packages/pivot-tui/tests --timeout=60
      4. Assert: exit code 0
    Expected Result: All tests pass
    Evidence: pytest output captured

  Scenario: Quality checks pass
    Tool: Bash
    Preconditions: uv sync complete
    Steps:
      1. uv run ruff format --check .
      2. Assert: exit code 0
      3. uv run ruff check .
      4. Assert: exit code 0
      5. uv run basedpyright
      6. Assert: exit code 0
    Expected Result: No lint, format, or type errors
    Evidence: Tool outputs captured
  ```

  **Commit**: YES
  - Message: `refactor: complete monorepo repackaging of pivot as workspace member`
  - Files: `packages/pivot/pyproject.toml`, `pyproject.toml`, `packages/pivot-tui/pyproject.toml`, `mkdocs.yml`, `AGENTS.md`, `uv.lock`, and any sub-AGENTS.md files updated
  - Pre-commit: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto && uv run ruff check . && uv run basedpyright`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 6 (all) | `refactor: complete monorepo repackaging of pivot as workspace member` | All changed files | pytest + ruff + basedpyright |

---

## Success Criteria

### Verification Commands
```bash
uv lock                                                              # Expected: exit 0
uv sync --active                                                     # Expected: exit 0
uv run pivot --help                                                  # Expected: shows CLI help
uv run pytest packages/pivot/tests -n auto                           # Expected: all pass
uv run pytest packages/pivot-tui/tests                               # Expected: all pass
uv run ruff format --check .                                         # Expected: exit 0
uv run ruff check .                                                  # Expected: exit 0
uv run basedpyright                                                  # Expected: exit 0
```

### Final Checklist
- [x] Both packages have pyproject.toml with uv_build backend
- [x] Root pyproject.toml is workspace-only (no runtime deps, no scripts)
- [x] Dev deps declared as extras on each package
- [x] Root dev dependency-group installs both packages with dev extras
- [x] All tool configs reference `packages/` paths
- [x] No stale `src/pivot/` or bare `tests/` references in config
- [x] uv.lock points to `packages/pivot` not `.` for pivot
- [x] All tests pass
- [x] All quality checks pass
- [x] CLI entry point works
