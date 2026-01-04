# Pivot: High-Performance Python Pipeline Tool

**Status:** In Development (Week 1 - Core Infrastructure)
**Python Version:** 3.13+ (3.14+ for experimental InterpreterPoolExecutor)
**Updated:** 2026-01-04

---

## Overview

Pivot is a Python-native pipeline tool designed to eliminate DVC's performance bottlenecks through:

1. **Per-stage lock files** - 32x faster than DVC's monolithic lock file
2. **Automatic code fingerprinting** - No manual dependency declarations needed
3. **Warm worker pool** - Preloaded imports eliminate startup overhead
4. **DVC compatibility** - Export to dvc.yaml for code review and validation

## Core Design Principles

### 1. Test-Driven Development (TDD)

- Write tests BEFORE implementation
- Each module has corresponding test file
- Integration tests validate end-to-end behavior
- Minimum 90% code coverage

### 2. Code Quality Standards

- **No code duplication** - Extract shared logic
- **Type hints everywhere** - Use basedpyright for type checking
- **Formatting** - Use ruff format (line length 100)
- **Linting** - Use ruff check for fast linting
- **Docstrings** - Concise, one-line preferred; avoid verbose documentation for simple helpers
- **Private functions** - Use leading underscore for module-internal functions
- **Import style** - Follow Google Python Style Guide: import modules, not functions/classes

### Python 3.13+ Type Hints (CRITICAL)

- **Use constructor syntax for empty collections** - `list[int]()` instead of `: list[int] = []`

  ```python
  # Good - Python 3.13+ constructor syntax
  self.items = list[int]()
  self.mapping = dict[str, int]()
  self.unique = set[str]()

  # Bad - old-style annotation with empty literal
  self.items: list[int] = []
  self.mapping: dict[str, int] = {}
  self.unique: set[str] = set()
  ```

- **Simplified Generator types** - Python 3.13+ infers default `None, None` for send/return types

  ```python
  from collections.abc import Generator

  # Good - Python 3.13+ simplified syntax
  def my_generator() -> Generator[int]:
      yield 1
      yield 2

  # Bad - unnecessary explicit None, None
  def my_generator() -> Generator[int, None, None]:
      yield 1
      yield 2

  # Note: Only include send/return types when non-None
  def with_send() -> Generator[int, str, None]:
      received = yield 1
      # ...
  ```

- **Use `Callable` instead of `Any` for callables** - Be specific about function types

  ```python
  from collections.abc import Callable

  # Good - specific callable signature
  def apply(func: Callable[[int], str], value: int) -> str:
      return func(value)

  # Bad - using Any for a callable
  def apply(func: Any, value: int) -> str:
      return func(value)
  ```

- **Use `Any` as last resort only** - Prefer specific types or protocols

  ```python
  # Good - specific type
  def process(data: dict[str, int]) -> int:
      return sum(data.values())

  # Good - protocol when structure is flexible
  from typing import Protocol

  class HasValue(Protocol):
      value: int

  def get_value(obj: HasValue) -> int:
      return obj.value

  # Bad - unnecessary Any
  def process(data: Any) -> Any:
      return sum(data.values())

  # OK - Any is truly needed (dynamic introspection)
  def inspect_function(func: Any) -> dict[str, str]:
      # Working with function objects dynamically via inspect module
      return {"name": func.__name__, "module": func.__module__}
  ```

- **When `Any` is acceptable**:
  - Dynamic introspection (inspect module usage)
  - Working with truly dynamic data (JSON, YAML parsing before validation)
  - Temporary placeholder during development (should be replaced before commit)
  - Testing code that intentionally tests with various types

### 3. Documentation and Comments

- **CLAUDE.md** at each level explaining intent
- **Docstrings** for public APIs (concise, one-line preferred)
- **Comments** for WHY, not WHAT
  - Good: `# Use forkserver to avoid GIL issues in forked processes`
  - Bad: `# Parse to AST` (obvious from `ast.parse()` call)
  - Bad: `# Loop through items` (obvious from `for item in items:`)
- **No redundant comments** - code should be self-documenting through naming
- Update docs when implementation diverges from plan
- README for user-facing documentation

---

## Development Tools

### uv - Package Management

**Purpose:** Fast Python package installer and virtual environment manager

**Setup:**

```bash
# Create virtual environment and sync dependencies
cd /workspaces/treeverse/pivot
uv venv
uv sync

# Activate virtual environment (optional - uv commands work without activation)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

**Usage:**

```bash
# Sync dependencies from pyproject.toml
uv sync

# Sync with dev dependencies (default)
uv sync --group dev

# Add a new dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Add specific version
uv add "numpy>=2.0"

# Remove a dependency
uv remove <package>

# Upgrade dependencies
uv sync --upgrade

# List installed packages
uv pip list

# Run a command in the virtual environment (no activation needed)
uv run pytest tests/
uv run ruff check src/
```

**Why uv?** 10-100x faster than pip, written in Rust, handles both venv and dependencies

### uv Build Backend (uv_build)

**Purpose:** Native uv build backend - fastest, zero-config Python package building

**Configuration:** `pyproject.toml`

```toml
[build-system]
requires = ["uv_build>=0.9.18,<0.10.0"]
build-backend = "uv_build"
```

**Building:**

```bash
# Build wheel and sdist
uv build

# Build only wheel
uv build --wheel

# Build only sdist
uv build --sdist

# Output will be in dist/
```

**Why uv_build?**

- Native uv integration - fastest possible builds
- Zero configuration needed (auto-detects src-layout)
- Standards-compliant (PEP 517/518/621)
- Better than hatchling/setuptools for uv-based projects
- See: https://docs.astral.sh/uv/concepts/build-backend/

**Project Structure:**

```
pivot/
├── pyproject.toml        # Build config + dependencies
├── src/
│   └── pivot/        # Package code (src-layout)
│       ├── __init__.py
│       └── ...
└── dist/                 # Build outputs (git-ignored)
    ├── pivot-0.1.0-py3-none-any.whl
    └── pivot-0.1.0.tar.gz
```

**Installation:**

```bash
# Install from local wheel (for testing)
uv pip install dist/pivot-0.1.0-py3-none-any.whl

# Install in editable mode (for development)
uv pip install -e .
```

---

### ruff - Linting and Formatting

**Purpose:** Fast Python linter and formatter (replaces black, flake8, isort)

**Linting:**

```bash
# Check all issues
ruff check .

# Check with file paths
ruff check . --show-files

# Auto-fix safe issues
ruff check . --fix

# Show rule explanations
ruff rule E501
```

**Formatting:**

```bash
# Check formatting (dry-run)
ruff format . --check

# Format files
ruff format .

# Format specific file
ruff format src/pivot/fingerprint.py
```

**Configuration:** See `pyproject.toml` [tool.ruff] section

**Common Rules:**

- `E` - pycodestyle errors (PEP 8 violations)
- `F` - pyflakes (unused imports, undefined names)
- `I` - isort (import sorting)
- `N` - pep8-naming (naming conventions)
- `UP` - pyupgrade (modern Python syntax)
- `B` - flake8-bugbear (common bugs)

**Pre-commit Workflow:**

```bash
# Before committing
ruff check . --fix
ruff format .
```

---

### basedpyright - Type Checking

**Purpose:** Fast Python type checker (fork of pyright with additional features)

**Usage:**

```bash
# Check all files
basedpyright

# Check specific directory
basedpyright src/

# Check specific file
basedpyright src/pivot/fingerprint.py

# Show configuration
basedpyright --version
```

**Configuration:** See `pyproject.toml` [tool.basedpyright] section

**Type Checking Mode:**

- `standard` - Balance of strictness and usability (our default)
- `basic` - Minimal type checking
- `strict` - Maximum type checking (consider for critical modules)

**Common Issues:**

```python
# Issue: Missing type hints
def process(data):  # Error: Missing parameter type
    return data

# Fix: Add type hints
def process(data: list[int]) -> list[int]:
    return data

# Issue: Unknown type
result = some_function()  # Type unknown
x = result.value  # Error: Unknown attribute

# Fix: Add type annotation
result: MyType = some_function()
x = result.value  # OK
```

**Workflow:**

```bash
# Check types during development
basedpyright src/pivot/fingerprint.py

# Fix issues, then re-check
basedpyright
```

---

### pytest - Testing Framework

**Purpose:** Modern Python testing framework

**Usage:**

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_fingerprint.py

# Run specific test function
pytest tests/test_fingerprint.py::test_simple_function_fingerprinted

# Run specific test class
pytest tests/test_fingerprint.py::TestGetStageFingerprint

# Run tests matching pattern
pytest -k "fingerprint"

# Run with coverage
pytest --cov=src/pivot --cov-report=term-missing

# Run with HTML coverage report
pytest --cov=src/pivot --cov-report=html

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Enter debugger on failure
pytest --pdb

# Re-run failed tests
pytest --lf

# Watch mode (re-run on changes)
pytest-watch tests/ -- -v
```

**Configuration:** See `pyproject.toml` [tool.pytest.ini_options] section

**Test Organization:**

- `tests/test_<module>.py` - Unit tests mirroring src/ structure
- `tests/integration/` - End-to-end integration tests
- `tests/conftest.py` - Shared fixtures

**Writing Tests:**

```python
import pytest
from pivot import fingerprint


def test_basic_behavior():
    """Test description."""
    result = fingerprint.get_stage_fingerprint(lambda: 42)
    assert "self:" in result


@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    """Will implement in Week 2."""
    pass


@pytest.fixture
def sample_function():
    """Reusable test fixture."""
    def func():
        return 42
    return func


def test_with_fixture(sample_function):
    """Use fixture in test."""
    assert sample_function() == 42
```

**Test-Driven Development (TDD) Workflow:**

```bash
# 1. Write test first (should fail)
# Edit tests/test_fingerprint.py
pytest tests/test_fingerprint.py::test_new_feature  # Fails ✗

# 2. Implement minimal code to pass
# Edit src/pivot/fingerprint.py
pytest tests/test_fingerprint.py::test_new_feature  # Passes ✓

# 3. Refactor while keeping tests green
# Edit src/pivot/fingerprint.py
pytest tests/test_fingerprint.py  # All pass ✓

# 4. Check coverage
pytest tests/test_fingerprint.py --cov=src/pivot/fingerprint --cov-report=term-missing
```

---

### Complete Development Workflow

**Initial Setup:**

```bash
cd /workspaces/treeverse/pivot
uv venv
uv sync

# Optional: activate venv (or use `uv run` for commands)
source .venv/bin/activate
```

**Before Starting Work:**

```bash
# Make sure tests pass
pytest tests/

# Check types
basedpyright

# Check linting
ruff check .
```

**During Development (TDD):**

```bash
# 1. Write test first
# Edit tests/test_<module>.py

# 2. Run test (should fail)
pytest tests/test_<module>.py::test_new_feature -v

# 3. Implement feature
# Edit src/pivot/<module>.py

# 4. Run test (should pass)
pytest tests/test_<module>.py::test_new_feature -v

# 5. Check types and format
basedpyright src/pivot/<module>.py
ruff format src/pivot/<module>.py

# 6. Run all tests for that module
pytest tests/test_<module>.py -v
```

**Before Committing:**

```bash
# Format code
ruff format .

# Fix linting issues
ruff check . --fix

# Check types
basedpyright

# Run all tests with coverage
pytest tests/ --cov=src/pivot --cov-report=term-missing

# Ensure coverage >= 90%
# If coverage too low, add more tests
```

**🚨 CRITICAL: Before Returning to User 🚨**

**Make sure you've cleaned up all errors before returning to the user:**

```bash
# 1. Format code (required)
uv run ruff format .

# 2. Check linting (required)
uv run ruff check .

# 3. Type check (required)
uv run basedpyright .

# 4. Run tests (required)
uv run pytest tests/
```

**Why this matters:**

- User expects code to be properly formatted and type-checked
- Catching errors before user sees them is professional
- Tests validate that changes don't break existing functionality
- Running these tools is part of the development workflow, not optional

**Never say you're "done" without running all four commands first.**

**Daily Development Commands:**

```bash
# Quick validation (fast)
ruff check . --fix && ruff format . && pytest tests/test_*.py

# Full validation (slower)
ruff check . && ruff format . --check && basedpyright && pytest tests/ --cov=src/pivot --cov-report=term-missing
```

---

## Architecture Overview

```
Pivot Architecture (Simplified)

┌─────────────────────────────────────────────────────────────┐
│  User Pipeline Code (Python decorators)                     │
│  @stage(deps=['data.csv'], outs=['model.pkl'])              │
│  def train(lr: float = 0.01): ...                          │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage Registry                                              │
│  - Collects decorated functions                             │
│  - Introspects signatures for params                        │
│  - Generates fingerprints (getclosurevars + AST)           │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  DAG Builder (NetworkX)                                      │
│  - Builds dependency graph from deps/outs                   │
│  - Topological sorting                                       │
│  - Cycle detection                                           │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Scheduler (Parallel or Sequential)                          │
│  - Ready queue pattern (adapted from DVC)                   │
│  - StageInfo tracking (upstream_unfinished)                 │
│  - FIRST_COMPLETED feedback loop                            │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Executor (Warm Workers or Interpreters)                    │
│  - WarmWorkerPoolExecutor (default)                         │
│  - InterpreterPoolExecutor (Python 3.14+, experimental)     │
│  - ProcessPoolExecutor (fallback)                           │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Lock Files (.pivot/stages/<name>.lock)                  │
│  - Per-stage YAML with code manifest                        │
│  - Parallel writes (no contention)                          │
│  - Fast CSafeLoader parsing                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Decision 1: Per-Stage Lock Files (vs DVC's Monolithic)

**Problem:** DVC writes entire dvc.lock file for each stage (O(n²) behavior)

- 176 stages × 1.6s/write = 282 seconds of lock file overhead
- Thread lock serializes writes even in parallel execution

**Solution:** Each stage gets its own lock file

- `.pivot/stages/train.lock` (~500 bytes - 2KB)
- Parallel writes without contention
- Expected: 176 stages × 0.05s = 9 seconds (32x improvement)

**Trade-off:** More files, but modern filesystems handle this well

### Decision 2: Automatic Code Fingerprinting (getclosurevars + AST)

**Problem:** DVC requires manual dependency declaration for Python code

**Solution:** Automatic detection using Python introspection

1. `inspect.getclosurevars()` captures referenced functions/constants
2. AST extraction for `module.attr` patterns (Google-style imports)
3. Recursive fingerprinting for transitive dependencies

**Validated:** `/workspaces/treeverse/pivot/experiments/test_getclosurevars_approach.py`

**Known Limitations:**

- Dynamic dispatch: `getattr(obj, variable_name)` - provide escape hatch
- Method calls: `obj.method()` - document limitation
- `eval()`/`exec()` - document limitation

### Decision 3: Warm Workers (Default) + Interpreters (Experimental)

**Problem:** Import overhead (numpy/pandas can take seconds)

**Solution 1 (Default):** Warm worker pool

- ProcessPoolExecutor with forkserver
- Preload numpy, pandas, scipy in worker processes
- Imports persist across stage executions

**Solution 2 (Experimental):** InterpreterPoolExecutor (Python 3.14+)

- True parallelism without GIL
- Faster startup (~0.024s vs fork overhead)
- Can't share preloaded modules (each interpreter has own import state)
- Lower memory footprint (30-50% vs processes)

**Configuration:**

```bash
# Default
pivot run --executor=default

# Experimental (Python 3.14+)
pivot run --executor=interpreter

# Environment variable
export PIVOT_EXECUTOR=interpreter
```

### Decision 4: DVC YAML Export for Code Review

**Problem:** Python decorators harder to review than static YAML

**Solution:** Export command generates dvc.yaml

```bash
pivot export  # Creates dvc.yaml with python -c commands
pivot export --validate  # Runs both Pivot and DVC, compares outputs
```

**Benefits:**

- PR reviews on familiar static YAML
- Validation tests ensure identical behavior
- Migration path from DVC
- Pipeline documentation

---

## Performance Targets (vs DVC)

Based on profiling monitoring-horizons pipeline (176 stages, 1214s total):

| Component         | DVC          | Pivot Target | Improvement |
| ----------------- | ------------ | ------------ | ----------- |
| Lock file writes  | 289s (23.8%) | 9s (0.7%)    | 32x         |
| Import overhead   | ~50-100s     | ~5-10s       | 10x         |
| Total overhead    | 301s (24.8%) | 20s (1.6%)   | 15x         |
| **Total runtime** | **1214s**    | **~950s**    | **1.3x**    |

---

## Development Roadmap

### Phase 1: MVP (Weeks 1-6) - IN PROGRESS

- [x] Planning and design validation
- [x] **Week 1: Fingerprinting + Registry** ✅ **COMPLETE**
  - 95 tests passing, 94.38% coverage
  - Automatic code fingerprinting working
  - Stage registry fully functional
- [ ] Week 2: DAG + Lock files + YAML export
- [ ] Week 3: Params + Hashing
- [ ] Week 4: Sequential executor + Explain mode
- [ ] Week 5: Parallel scheduler + Executors
- [ ] Week 6: CLI + End-to-end testing

### Phase 2: Observability (Weeks 7-9)

- Metrics tracking and diff
- Plots generation

### Phase 3: Version Control (Weeks 10-12)

- `pivot get --rev`
- DVC lock file import
- Git hooks

---

## Implementation Learnings

### Week 1: Fingerprinting + Registry (2026-01-04)

**What Worked Well:**

1. **TDD approach was highly effective**

   - Writing tests first caught design issues early
   - 95 tests provide excellent safety net for refactoring
   - Coverage target (90%) was realistic and achievable

2. **getclosurevars() + AST fingerprinting validated**

   - Top-level imports captured automatically (no lazy loading needed)
   - Transitive dependencies work recursively
   - AST normalization provides stable hashing
   - All experimental validations confirmed in production code

3. **Code quality standards**
   - Concise one-line docstrings keep code readable
   - "WHY not WHAT" comments improve maintainability
   - Module-level organization is clean and testable

**Critical Discoveries:**

1. **Test helpers must be module-level (not inline)**

   ```python
   # ❌ WRONG - inline function doesn't capture imports
   def test_math():
       def uses_math():
           return math.pi
       fp = get_stage_fingerprint(uses_math)
       assert "mod:math.pi" in fp  # FAILS!

   # ✅ CORRECT - module-level captures imports
   def helper_uses_math():  # At top of file
       return math.pi

   def test_math():
       fp = get_stage_fingerprint(helper_uses_math)
       assert "mod:math.pi" in fp  # WORKS!
   ```

   **Reason:** `inspect.getclosurevars()` doesn't see module-level imports in inline function closures.

2. **Helper functions must NOT start with underscore**

   - Fingerprinting filters `_globals` to skip `__name__`, `__file__`, etc.
   - This also filters user functions like `_helper()`
   - **Solution:** Use `helper_` prefix instead of `_helper_`

3. **Stage decorator dataclass pattern**

   ```python
   @dataclass
   class stage:
       deps: list[str] = field(default_factory=list)
       outs: list[str] = field(default_factory=list)

       def __call__(self, func):
           REGISTRY.register(func, deps=self.deps, outs=self.outs)
           return func
   ```

   - Clean syntax: `@stage(deps=[...])`
   - Default values via dataclass fields
   - Type safety built-in

4. **Assertion messages beat inline comments**

   ```python
   # ✅ GOOD - message appears in failure output
   assert "func:helper" in fp, "Should capture helper function"

   # ❌ BAD - comment doesn't help when test fails
   # Should capture helper function
   assert "func:helper" in fp
   ```

**Files Created:**

- `src/pivot/fingerprint.py` (201 lines) - Core fingerprinting
- `src/pivot/ast_utils.py` (85 lines) - AST utilities
- `src/pivot/registry.py` (99 lines) - Stage registration
- `tests/test_fingerprint.py` (904 lines) - 70 tests
- `tests/test_registry.py` (281 lines) - 20 tests
- `tests/test_change_detection.py` - Integration tests
- `tests/test_google_style.py` - Import validation

**Coverage Achieved:** 94.38% (target: 90%)

- `fingerprint.py`: 93.64%
- `ast_utils.py`: 93.75%
- `registry.py`: 100% ✨
- `__init__.py`: 100%

**Blockers Resolved:**

- ✅ Registry implementation complete (was blocking Week 2)
- ✅ All tests passing
- ✅ Coverage above target
- ✅ Type safety verified (0 errors)

**Ready for Week 2:** All Week 1 deliverables complete and validated.

---

## Testing Strategy

### Unit Tests

- Location: `tests/test_<module>.py`
- Coverage: >90%
- Run: `pytest tests/test_*.py -v`

### Integration Tests

- Location: `tests/integration/`
- Real pipeline patterns
- DVC compatibility validation

### Performance Tests

- Location: `benchmarks/`
- Compare against DVC
- Memory/CPU profiling

---

## Code Organization

See `pivot/src/CLAUDE.md` for detailed module documentation.

```
pivot/
├── CLAUDE.md                 # This file - high-level design
├── README.md                 # User-facing documentation
├── pyproject.toml            # Dependencies and build config
├── src/pivot/             # Source code
│   ├── CLAUDE.md            # Module-level design docs
│   ├── __init__.py          # Public API
│   ├── fingerprint.py       # Code change detection
│   ├── registry.py          # Stage collection
│   ├── dag.py               # Dependency graph
│   ├── ... (see src/CLAUDE.md for full list)
├── tests/                    # Unit tests (mirrors src/)
│   ├── CLAUDE.md            # Testing strategy
│   ├── test_fingerprint.py
│   ├── ...
├── benchmarks/               # Performance tests
├── experiments/              # Research and validation
└── docs/                     # Additional documentation
```

---

## Key References

### DVC Implementation (Study These)

- `/workspaces/treeverse/dvc/dvc/repo/reproduce.py:145-283` - Parallel execution pattern
- `/workspaces/treeverse/dvc/dvc/dvcfile.py:444-473` - Lock file writing bottleneck
- `/workspaces/treeverse/dvc/dvc/repo/graph.py` - DAG operations

### Profiling Data (Validates Design)

- `/workspaces/treeverse/timing/TIMING_REPORT.md` - 60 stages, 50s lock overhead
- `/workspaces/treeverse/timing/MONITORING_HORIZONS_TIMING_REPORT.md` - 176 stages, 289s lock overhead

### Experimental Validation

- `/workspaces/treeverse/pivot/experiments/test_getclosurevars_approach.py` - Fingerprinting proof

---

## Assumptions and Constraints

### Assumptions

1. **Python 3.13+ available** - Need modern typing and performance
2. **Linux/Mac primary** - forkserver context (Windows fallback: spawn)
3. **Git repository** - For version tracking features (Phase 3)
4. **Filesystem performance** - Per-stage locks assume fast filesystem (SSDs)

### Constraints

1. **No breaking changes to user pipelines** - Export to dvc.yaml maintains compatibility
2. **DVC coexistence** - Can run alongside DVC during migration
3. **Type safety** - All public APIs must be fully typed
4. **Backward compatibility** - Later versions must read earlier lock files

---

## Questions and Decisions Log

### Open Questions

1. Should we support matrix stages (DVC's `foreach`) in MVP? → DECIDE
2. How to handle stages with side effects (API calls, DB writes)? → DECIDE
3. Should lock files be git-tracked or gitignored? → DECIDE

### Resolved Decisions

- ✅ Use warm workers as default (proven), interpreters as experimental
- ✅ Export to dvc.yaml for code review
- ✅ Per-stage lock files (not batched)
- ✅ getclosurevars() + AST for fingerprinting

---

**Next Steps:** See `src/CLAUDE.md` for module-by-module implementation plan.
