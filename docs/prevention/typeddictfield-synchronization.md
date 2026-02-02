# TypedDict Field Synchronization Prevention

## Problem Pattern

When a TypedDict gains a new **required field**, all test code creating instances of that TypedDict must be updated simultaneously. basedpyright catches this during type checking, but only if run locally before pushing.

### Example: Breaking Change

```python
# Before: PipelineReloaded has 5 required fields
class PipelineReloaded(TypedDict):
    type: Literal["pipeline_reloaded"]
    stages: list[str]
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None

# After: Adding a new required field breaks all test instances
class PipelineReloaded(TypedDict):
    type: Literal["pipeline_reloaded"]
    stages: list[str]
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None
    is_critical: bool  # NEW REQUIRED FIELD - tests now fail type checking
```

Tests that create instances without `is_critical` now fail:

```python
# This now violates the TypedDict contract
event: types.PipelineReloaded = {
    "type": "pipeline_reloaded",
    "stages": ["new_stage"],
    "stages_added": ["new_stage"],
    "stages_removed": [],
    "stages_modified": [],
    "error": None,
    # Missing: is_critical
}
```

### Why This Breaks in CI

1. Local development may not run full type checking before commit
2. Developer only tests happy path, misses type errors in other test files
3. CI runs comprehensive type checking with `basedpyright`, catching all violations
4. Multiple test files creating the same TypedDict type must be updated atomically

---

## Prevention Strategies

### 1. Always Run Local Type Checking (Critical)

**Run before every push:**

```bash
# Full quality check suite - required before pushing
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

This can be aliased or added to pre-commit hooks:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias check-pivot="cd ~/pivot/default && uv run ruff format . && uv run ruff check . && uv run basedpyright ."
```

**Frequency:**
- After modifying any TypedDict definition
- After modifying any type hints
- Before committing any changes
- Before pushing to remote

### 2. Prefer NotRequired for Optional Fields (Design Prevention)

Avoid adding new required fields to existing TypedDicts. Instead, use `NotRequired`:

```python
# Good: New fields won't break existing code
class PipelineReloaded(TypedDict, total=False):
    type: Literal["pipeline_reloaded"]
    stages: Required[list[str]]  # Required fields explicit
    stages_added: Required[list[str]]
    stages_removed: Required[list[str]]
    stages_modified: Required[list[str]]
    error: Required[str | None]
    is_critical: NotRequired[bool]  # Non-breaking

# Better: If truly required, add to a separate TypedDict version
class PipelineReloadedV2(TypedDict):
    """Next API version with additional required field."""
    type: Literal["pipeline_reloaded"]
    stages: list[str]
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None
    is_critical: bool  # Required, new version only
```

**Principle:** Pre-alpha projects can break compatibility, but do so intentionally with versioned types, not by silently requiring new fields.

### 3. Search for All Usages Before Adding Required Fields

When a TypedDict **must** gain a required field, find all existing usages first:

```bash
# Find all test files creating instances of PipelineReloaded
grep -r "PipelineReloaded\|pipeline_reloaded" tests/ --include="*.py"

# Results show all files needing updates:
# - tests/engine/test_types.py
# - tests/engine/test_engine.py
# - tests/engine/test_sinks.py
# - tests/integration/test_unified_execution.py
```

**Before modifying the TypedDict:**

1. Run the grep search above
2. Open each file and verify how it creates instances
3. Plan updates for all instances simultaneously
4. Make TypedDict change + all test updates in one commit

### 4. Use TypedDict Constructor Syntax (IDE Support)

Use explicit constructor syntax instead of dict literals. This provides better IDE hints and error messages:

```python
# Less IDE support - errors less obvious
event: types.PipelineReloaded = {
    "type": "pipeline_reloaded",
    # ... missing field won't be highlighted until type check
}

# Better IDE support - missing fields highlighted as you type
event: types.PipelineReloaded = types.PipelineReloaded(
    type="pipeline_reloaded",
    stages=["new"],
    stages_added=["new"],
    stages_removed=[],
    stages_modified=[],
    error=None,
)
```

**Note:** TypedDict constructor syntax requires Python 3.12+ (this project uses 3.13+).

**Benefit:** IDEs will immediately show required fields, catching omissions before type checking.

### 5. Run Full Test Suite Including Type Checks

**Complete quality check before push:**

```bash
# 1. Format code
uv run ruff format .

# 2. Check style and errors
uv run ruff check .

# 3. Run type checks (CRITICAL for TypedDict fields)
uv run basedpyright .

# 4. Run tests with coverage
uv run pytest tests/ -n auto --cov=src/pivot --cov-fail-under=90
```

Or create a pre-commit hook:

```bash
#!/bin/bash
# .git/hooks/pre-commit (make executable: chmod +x)

set -e

echo "Running quality checks..."
uv run ruff format . >/dev/null 2>&1
uv run ruff check . >/dev/null 2>&1
uv run basedpyright . >/dev/null 2>&1

echo "✓ All checks passed"
```

### 6. Enable Strict TypedDict Mode in IDE

**For VS Code:**

```json
{
  "[python]": {
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": false
  },
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "basedpyright.analysis.typeCheckingMode": "strict"
}
```

This ensures TypedDict violations appear in the editor, not just CI.

### 7. Document TypedDict Contracts in Test Files

Add comments to test files documenting required fields:

```python
def test_pipeline_reloaded_event() -> None:
    """PipelineReloaded event has required fields.

    REQUIRED FIELDS (must be present in all instances):
    - type: Literal["pipeline_reloaded"]
    - stages: list[str]
    - stages_added: list[str]
    - stages_removed: list[str]
    - stages_modified: list[str]
    - error: str | None

    When adding new required fields to PipelineReloaded, search for
    all usages: grep -r "PipelineReloaded" tests/
    """
    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages": ["new_stage"],
        "stages_added": ["new_stage"],
        "stages_removed": [],
        "stages_modified": [],
        "error": None,
    }
    assert event["type"] == "pipeline_reloaded"
```

### 8. Group TypedDict Definitions by Stability

Keep stable TypedDicts separate from evolving ones:

```python
# Stable TypedDicts - rarely change
class FileHash(TypedDict):
    hash: str

class DirManifestEntry(TypedDict):
    relpath: str
    hash: str
    size: int
    isexec: bool

# Evolving TypedDicts - API surface area, expect changes
class RunRequested(TypedDict):
    type: Literal["run_requested"]
    stages: list[str] | None
    force: bool
    reason: str
    # ... more fields added with NotRequired
```

**Benefit:** Developers can quickly identify which types are changing frequently and need extra attention.

### 9. Create Type-Checking CI Step

Ensure CI explicitly runs type checking as a separate, early step:

```yaml
# .github/workflows/ci.yml (example)
jobs:
  types:
    name: Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v2
      - name: Run basedpyright
        run: uv run basedpyright .

  tests:
    name: Tests
    runs-on: ubuntu-latest
    needs: types  # Run type check first
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: uv run pytest tests/ -n auto
```

This makes type errors visible before test execution.

---

## Workflow Summary

### When Adding a Required Field to an Existing TypedDict

1. **Search for all usages:**
   ```bash
   grep -r "TypedDictName" tests/ --include="*.py"
   ```

2. **Plan changes across all files** - don't commit TypedDict change until all usages identified

3. **Update TypedDict and all test usages in one commit:**
   ```bash
   # Edit src/pivot/engine/types.py (add required field)
   # Edit tests/engine/test_types.py (add field to instances)
   # Edit tests/engine/test_engine.py (add field to instances)
   # ... (repeat for all files found in search)
   ```

4. **Run full quality checks locally:**
   ```bash
   uv run ruff format . && uv run ruff check . && uv run basedpyright .
   ```

5. **Push when all checks pass:**
   ```bash
   jj git push  # or git push
   ```

### When Designing New TypedDicts

- Use `NotRequired` for fields that may be absent or added in future versions
- Use `Required[]` explicitly when using `total=False` with mandatory fields
- Prefer TypedDict constructor syntax in tests for IDE support
- Add docstrings listing required fields

### Before Every Push

```bash
# Mandatory quality checks
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Find all usages of a TypedDict | `grep -r "TypedDictName" tests/` |
| Type check locally | `uv run basedpyright .` |
| Full quality check | `uv run ruff format . && uv run ruff check . && uv run basedpyright .` |
| Run tests with type checks | `uv run pytest tests/ -n auto --co` (shows collection phase errors) |
| Enable IDE type checking | Set `basedpyright.analysis.typeCheckingMode: strict` in VS Code settings |

---

## Why This Matters

- **Atomic updates:** TypedDict changes + all test updates in one commit prevents partial breakage
- **Early detection:** Local type checking catches errors before CI
- **IDE support:** Constructor syntax and strict mode catch errors as you type
- **Documentation:** Explicit field listings help reviewers verify completeness
- **Pre-alpha confidence:** Even though breaking changes are allowed, systematic prevention builds confidence in the codebase

